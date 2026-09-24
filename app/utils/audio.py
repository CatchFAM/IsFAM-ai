import importlib
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from fastapi import UploadFile

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """Base exception for upload/audio validation errors."""


class MissingAudioFileError(AudioValidationError):
    """Raised when a required upload file is missing."""


class UnsupportedAudioFormatError(AudioValidationError):
    """Raised when the upload extension is not allowed."""


class UploadFileTooLargeError(AudioValidationError):
    """Raised when an upload exceeds the configured size limit."""


class AudioDecodingError(AudioValidationError):
    """Raised when torchaudio cannot decode the uploaded file."""


class AudioTooShortError(AudioValidationError):
    """Raised when the decoded audio is shorter than the minimum duration."""


def _import_torchaudio() -> Any:
    try:
        return importlib.import_module("torchaudio")
    except Exception as exc:
        raise AudioDecodingError(
            "torchaudio is not installed or cannot be loaded. Run: pip install -r requirements.txt"
        ) from exc


def _find_ffmpeg() -> str | None:
    """Find FFmpeg even when the server process has a limited PATH."""

    candidates = (
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew default
        "/usr/local/bin/ffmpeg",  # Intel macOS Homebrew default
        "/usr/bin/ffmpeg",
    )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate

    return None


def validate_audio_extension(filename: str | None, allowed_extensions: Iterable[str]) -> str:
    """Validate filename extension and return it without a dot.

    The API accepts wav/mp3/m4a by default. Actual decoding support depends on
    the local torchaudio/FFmpeg backend installed in the runtime environment.
    """

    if not filename:
        raise MissingAudioFileError("audio filename is missing")

    suffix = Path(filename).suffix.lower().lstrip(".")
    allowed = {extension.lower().lstrip(".") for extension in allowed_extensions}

    if suffix not in allowed:
        raise UnsupportedAudioFormatError(
            f"unsupported audio format: .{suffix or 'unknown'}"
        )

    return suffix


async def save_upload_file_to_temp(
    upload_file: UploadFile,
    allowed_extensions: Iterable[str],
    max_size_bytes: int,
) -> Path:
    """Save an uploaded file to a temporary file while enforcing size limit."""

    extension = validate_audio_extension(upload_file.filename, allowed_extensions)
    temp_path = Path(tempfile.gettempdir()) / f"isfam_{uuid4().hex}.{extension}"

    total_size = 0
    chunk_size = 1024 * 1024

    try:
        with temp_path.open("wb") as temp_file:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > max_size_bytes:
                    raise UploadFileTooLargeError(
                        f"uploaded file exceeds max size: {max_size_bytes} bytes"
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise MissingAudioFileError("uploaded file is empty")

        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()


def load_audio_as_mono_16k(
    input_path: Path,
    target_sample_rate: int,
    min_audio_seconds: float,
) -> tuple[Any, int, float]:
    """Decode audio, convert it to mono, resample it, and validate duration.

    Returns:
        waveform: Tensor shaped [1, time]
        sample_rate: target sample rate
        duration_seconds: decoded duration after resampling
    """

    torchaudio = _import_torchaudio()

    try:
        waveform, sample_rate = torchaudio.load(str(input_path))
    except Exception as exc:
        logger.info("torchaudio decode failed for %s: %s", input_path, exc)
        raise AudioDecodingError("failed to decode audio with torchaudio") from exc

    if waveform.numel() == 0:
        raise AudioDecodingError("decoded audio is empty")

    # torchaudio returns [channels, time]. Average channels to make mono.
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    duration_seconds = waveform.shape[-1] / float(target_sample_rate)
    if duration_seconds < min_audio_seconds:
        raise AudioTooShortError(
            f"audio is too short: {duration_seconds:.2f}s. "
            f"minimum is {min_audio_seconds:.2f}s."
        )

    return waveform.contiguous(), target_sample_rate, duration_seconds


def convert_audio_with_ffmpeg(
    input_path: Path,
    target_sample_rate: int,
    min_audio_seconds: float,
) -> Path:
    """Use FFmpeg CLI to convert mp3/m4a/etc. to 16kHz mono wav.

    Some torchaudio wheels on macOS cannot decode m4a directly even when the
    ffmpeg command is installed. This fallback uses the ffmpeg executable first,
    then validates the generated wav with Python's standard wave module.
    """

    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path is None:
        raise AudioDecodingError(
            "failed to decode audio with torchaudio, and ffmpeg executable was not found. "
            "Install it with: brew install ffmpeg"
        )

    output_path = Path(tempfile.gettempdir()) / f"isfam_{uuid4().hex}.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output_path.unlink(missing_ok=True)
        raise AudioDecodingError("ffmpeg audio conversion timed out") from exc
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise AudioDecodingError("failed to run ffmpeg audio conversion") from exc

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        stderr = completed.stderr.strip()
        detail = f": {stderr[:500]}" if stderr else ""
        raise AudioDecodingError(f"ffmpeg failed to decode audio{detail}")

    try:
        validate_standard_wav(
            wav_path=output_path,
            target_sample_rate=target_sample_rate,
            min_audio_seconds=min_audio_seconds,
        )
    except AudioValidationError:
        output_path.unlink(missing_ok=True)
        raise

    return output_path


def validate_standard_wav(
    wav_path: Path,
    target_sample_rate: int,
    min_audio_seconds: float,
) -> None:
    """Validate a generated PCM wav file without depending on torchaudio."""

    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except Exception as exc:
        raise AudioDecodingError("failed to validate converted wav") from exc

    if frame_count <= 0:
        raise AudioDecodingError("converted wav is empty")

    if sample_rate != target_sample_rate:
        raise AudioDecodingError(
            f"converted wav sample rate is {sample_rate}, expected {target_sample_rate}"
        )

    duration_seconds = frame_count / float(sample_rate)
    if duration_seconds < min_audio_seconds:
        raise AudioTooShortError(
            f"audio is too short: {duration_seconds:.2f}s. "
            f"minimum is {min_audio_seconds:.2f}s."
        )


def convert_audio_to_standard_wav(
    input_path: Path,
    target_sample_rate: int,
    min_audio_seconds: float,
) -> Path:
    """Convert uploaded audio to a temporary 16kHz mono wav file."""

    output_path = Path(tempfile.gettempdir()) / f"isfam_{uuid4().hex}.wav"
    try:
        waveform, sample_rate, _ = load_audio_as_mono_16k(
            input_path=input_path,
            target_sample_rate=target_sample_rate,
            min_audio_seconds=min_audio_seconds,
        )

        torchaudio = _import_torchaudio()
        torchaudio.save(
            str(output_path),
            waveform,
            sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        return output_path
    except AudioDecodingError:
        output_path.unlink(missing_ok=True)
        logger.info("torchaudio decode failed. Trying ffmpeg fallback for %s", input_path)
        return convert_audio_with_ffmpeg(
            input_path=input_path,
            target_sample_rate=target_sample_rate,
            min_audio_seconds=min_audio_seconds,
        )
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        logger.exception("Failed to write normalized wav file: %s", output_path)
        raise AudioDecodingError("failed to convert audio to wav") from exc


def cleanup_temp_files(paths: Iterable[Path | None]) -> bool:
    """Delete temporary files and report whether every deletion succeeded."""

    all_purged = True
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except Exception:
            all_purged = False
            logger.warning("Failed to delete temporary file: %s", path, exc_info=True)
    return all_purged
