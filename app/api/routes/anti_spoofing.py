import logging
import asyncio
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.schemas.anti_spoofing import (
    AntiSpoofingAudioQuality,
    AntiSpoofingLabelScore,
    AntiSpoofingModelInfoResponse,
    AntiSpoofingResponse,
)
from app.services.anti_spoofing_service import AntiSpoofingError, AntiSpoofingResult
from app.services.model_provider import get_anti_spoofing_service
from app.utils.audio import (
    AudioDecodingError,
    AudioTooShortError,
    AudioValidationError,
    MissingAudioFileError,
    UnsupportedAudioFormatError,
    UploadFileTooLargeError,
    cleanup_temp_files,
    convert_audio_to_standard_wav,
    save_upload_file_to_temp,
)
from app.utils.audio_quality import AudioQualityError, analyze_standard_wav_quality

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/anti-spoofing", tags=["anti-spoofing"])


@lru_cache(maxsize=1)
def get_anti_spoofing_semaphore() -> asyncio.Semaphore:
    return asyncio.Semaphore(get_settings().anti_spoofing_max_concurrency)


@router.get("/model-info", response_model=AntiSpoofingModelInfoResponse)
async def get_anti_spoofing_model_info(
    settings: Settings = Depends(get_settings),
) -> AntiSpoofingModelInfoResponse:
    """Return the active deepvoice model contract for clients and operations."""

    service = get_anti_spoofing_service()
    return AntiSpoofingModelInfoResponse(
        status="ready",
        model_name=service.model_name,
        model_version=settings.anti_spoofing_model_version,
        device=service.device,
        threshold=service.threshold,
        sample_rate=service.target_sample_rate,
        max_audio_seconds=service.max_audio_seconds,
        window_seconds=service.window_seconds,
        hop_seconds=service.hop_seconds,
        batch_size=service.batch_size,
        max_concurrency=settings.anti_spoofing_max_concurrency,
        warmed_up=service.is_warmed_up,
    )


def anti_spoofing_result_to_response(
    result: AntiSpoofingResult,
    *,
    processing_time_ms: float,
    audio_quality: AntiSpoofingAudioQuality,
) -> AntiSpoofingResponse:
    if not audio_quality.is_analyzable:
        analysis_status = "more_voice_required"
    elif result.message == "additional_confirmation":
        analysis_status = "additional_confirmation"
    else:
        analysis_status = "complete"
    return AntiSpoofingResponse(
        analysis_status=analysis_status,
        processing_time_ms=processing_time_ms,
        is_spoofed=result.is_spoofed,
        spoof_score=result.spoof_score,
        threshold=result.threshold,
        predicted_label=result.predicted_label,
        predicted_score=result.predicted_score,
        message=result.message,
        model_name=result.model_name,
        analyzed_segments=result.analyzed_segments,
        max_spoof_segment_index=result.max_spoof_segment_index,
        segment_seconds=result.segment_seconds,
        label_scores=[
            AntiSpoofingLabelScore(label=label_score.label, score=label_score.score)
            for label_score in result.label_scores
        ],
        audio_quality=audio_quality,
    )


@router.post("/detect", response_model=AntiSpoofingResponse)
async def detect_spoofed_voice(
    audio_file: UploadFile | None = File(
        default=None,
        description="Voice file to classify as bonafide or spoof/deepfake.",
    ),
    settings: Settings = Depends(get_settings),
) -> AntiSpoofingResponse:
    """Detect whether one uploaded voice looks spoofed/deepfake."""

    temp_paths: list[Path | None] = []

    try:
        if audio_file is None:
            raise MissingAudioFileError("audio_file is required")

        original_file = await save_upload_file_to_temp(
            upload_file=audio_file,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_file)

        wav_file = await run_in_threadpool(
            convert_audio_to_standard_wav,
            input_path=original_file,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_file)

        quality = await run_in_threadpool(
            analyze_standard_wav_quality,
            wav_path=wav_file,
            target_sample_rate=settings.target_sample_rate,
            min_analyzable_seconds=settings.voice_session_min_analyzable_seconds,
            min_rms_energy=settings.voice_session_min_rms_energy,
            min_speech_ratio=settings.voice_session_min_speech_ratio,
            min_estimated_snr_db=settings.voice_session_min_estimated_snr_db,
        )

        # Load the model only after the upload is valid and converted.
        inference_started_at = perf_counter()
        async with get_anti_spoofing_semaphore():
            result = await run_in_threadpool(
                get_anti_spoofing_service().detect_file,
                wav_file,
            )
        processing_time_ms = round((perf_counter() - inference_started_at) * 1000.0, 2)
        return anti_spoofing_result_to_response(
            result,
            processing_time_ms=processing_time_ms,
            audio_quality=AntiSpoofingAudioQuality(
                is_analyzable=quality.is_analyzable,
                message=quality.message,
                duration_seconds=quality.duration_seconds,
                rms_energy=quality.rms_energy,
                peak_amplitude=quality.peak_amplitude,
                speech_ratio=quality.speech_ratio,
                estimated_snr_db=quality.estimated_snr_db,
            ),
        )

    except MissingAudioFileError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except UnsupportedAudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except UploadFileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except AudioTooShortError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AudioQualityError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AntiSpoofingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during anti-spoofing detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during anti-spoofing detection",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)
