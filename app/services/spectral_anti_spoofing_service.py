import array
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.anti_spoofing_service import (
    AntiSpoofingError,
    AntiSpoofingResult,
    LabelScore,
)


SPECTRAL_FEATURE_COUNT = 196


def extract_spectral_features(
    samples: np.ndarray,
    sample_rate: int,
    max_seconds: float = 5.0,
) -> np.ndarray:
    """Extract inexpensive time/frequency statistics from one call chunk."""

    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    max_samples = max(1, int(sample_rate * max_seconds))
    if waveform.size > max_samples:
        start = (waveform.size - max_samples) // 2
        waveform = waveform[start : start + max_samples]

    frame_size = max(1, int(sample_rate * 0.025))
    hop_size = max(1, int(sample_rate * 0.010))
    if waveform.size < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.size))

    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_size)[::hop_size]
    windowed = frames * np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(windowed, n=512)) + 1e-7
    log_spectrum = np.log(spectrum)
    frequency_groups = np.array_split(np.arange(log_spectrum.shape[1]), 64)
    pooled = np.stack(
        [log_spectrum[:, group].mean(axis=1) for group in frequency_groups],
        axis=1,
    )
    deltas = np.diff(pooled, axis=0) if len(pooled) > 1 else np.zeros_like(pooled)

    zero_crossing_rate = (
        float(np.mean(waveform[:-1] * waveform[1:] < 0)) if waveform.size > 1 else 0.0
    )
    rms_energy = float(np.sqrt(np.mean(waveform * waveform)))
    peak_amplitude = float(np.max(np.abs(waveform)))
    duration_seconds = waveform.size / float(sample_rate)
    features = np.concatenate(
        [
            pooled.mean(axis=0),
            pooled.std(axis=0),
            deltas.std(axis=0),
            np.asarray(
                [zero_crossing_rate, rms_energy, peak_amplitude, duration_seconds],
                dtype=np.float64,
            ),
        ]
    ).astype(np.float32)
    if features.shape != (SPECTRAL_FEATURE_COUNT,):
        raise AntiSpoofingError(f"unexpected spectral feature shape: {features.shape}")
    return features


class SpectralAntiSpoofingService:
    """Small calibrated MLP for fast synthetic-speech screening on CPU."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.spectral_anti_spoofing_model_name
        self.target_sample_rate = settings.target_sample_rate
        self.window_seconds = settings.anti_spoofing_window_seconds
        self.max_audio_seconds = self.window_seconds
        self.hop_seconds = settings.anti_spoofing_hop_seconds
        self.batch_size = 1
        self.device = "cpu"
        self._is_warmed_up = False

        artifact_path = settings.spectral_anti_spoofing_model_path
        metadata_path = artifact_path.with_suffix(".json")
        if not artifact_path.exists() or not metadata_path.exists():
            raise AntiSpoofingError(
                f"spectral anti-spoof artifact is missing: {artifact_path}. "
                "Run scripts/train_spectral_anti_spoofing.py"
            )
        try:
            artifact = np.load(artifact_path, allow_pickle=False)
            self.feature_mean = artifact["feature_mean"].astype(np.float32)
            self.feature_std = artifact["feature_std"].astype(np.float32)
            self.hidden_weight = artifact["hidden_weight"].astype(np.float32)
            self.hidden_bias = artifact["hidden_bias"].astype(np.float32)
            self.output_weight = artifact["output_weight"].astype(np.float32)
            self.output_bias = artifact["output_bias"].astype(np.float32)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AntiSpoofingError("failed to load spectral anti-spoof artifact") from exc

        self.threshold = settings.anti_spoofing_threshold
        if self.feature_mean.shape != (SPECTRAL_FEATURE_COUNT,):
            raise AntiSpoofingError("spectral artifact feature count does not match runtime")
        if int(metadata.get("feature_count", -1)) != SPECTRAL_FEATURE_COUNT:
            raise AntiSpoofingError("spectral metadata feature count does not match runtime")

    def warm_up(self) -> None:
        if self._is_warmed_up:
            return
        self._predict_score(np.zeros(SPECTRAL_FEATURE_COUNT, dtype=np.float32))
        self._is_warmed_up = True

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def detect_file(self, wav_path: Path) -> AntiSpoofingResult:
        samples = self._load_standard_wav_samples(wav_path)
        features = extract_spectral_features(
            samples,
            sample_rate=self.target_sample_rate,
            max_seconds=self.window_seconds,
        )
        raw_score = self._predict_score(features)
        spoof_score = round(raw_score, 4)
        is_spoofed = raw_score >= self.threshold
        predicted_label = "fake" if raw_score >= 0.5 else "real"
        predicted_score = raw_score if raw_score >= 0.5 else 1.0 - raw_score
        return AntiSpoofingResult(
            is_spoofed=is_spoofed,
            spoof_score=spoof_score,
            threshold=round(self.threshold, 4),
            predicted_label=predicted_label,
            predicted_score=round(predicted_score, 4),
            message="spoof" if is_spoofed else "bonafide",
            model_name=self.model_name,
            analyzed_segments=1,
            max_spoof_segment_index=0,
            segment_seconds=self.window_seconds,
            label_scores=[
                LabelScore(label="real", score=round(1.0 - raw_score, 4)),
                LabelScore(label="fake", score=spoof_score),
            ],
        )

    def _predict_score(self, features: np.ndarray) -> float:
        normalized = (features - self.feature_mean) / self.feature_std
        hidden = normalized @ self.hidden_weight.T + self.hidden_bias
        # Match torch.nn.GELU(approximate="tanh") used by the training script.
        hidden = 0.5 * hidden * (
            1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden**3))
        )
        logit = float(hidden @ self.output_weight.reshape(-1) + self.output_bias.reshape(-1)[0])
        if logit >= 0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exp_logit = math.exp(logit)
        return float(exp_logit / (1.0 + exp_logit))

    def _load_standard_wav_samples(self, wav_path: Path) -> np.ndarray:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw_audio = wav_file.readframes(wav_file.getnframes())
        except Exception as exc:
            raise AntiSpoofingError("failed to read wav for spectral anti-spoofing") from exc
        if sample_rate != self.target_sample_rate or sample_width != 2 or not raw_audio:
            raise AntiSpoofingError("spectral anti-spoofing expects non-empty 16 kHz PCM16 wav")

        pcm = array.array("h")
        pcm.frombytes(raw_audio)
        if sys.byteorder == "big":
            pcm.byteswap()
        samples = np.asarray(pcm, dtype=np.float32)
        if channel_count > 1:
            samples = samples.reshape(-1, channel_count).mean(axis=1)
        return samples / 32768.0
