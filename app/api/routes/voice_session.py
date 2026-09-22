import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.repositories.family_repository import FamilyRepository
from app.repositories.voice_session_repository import VoiceSessionRepository
from app.schemas.anti_spoofing import AntiSpoofingLabelScore, AntiSpoofingResponse
from app.schemas.voice import FamilyCandidateResponse, VerifyFamilyResponse
from app.schemas.voice_session import (
    AudioQualityResponse,
    VoiceSessionChunkResponse,
    VoiceSessionStartResponse,
    VoiceSessionStatusResponse,
)
from app.services.anti_spoofing_service import AntiSpoofingError, AntiSpoofingResult
from app.services.model_provider import get_anti_spoofing_service, get_speaker_service
from app.services.speaker_service import SpeakerVerificationError
from app.services.voice_session_service import (
    VoiceSessionChunkAnalysis,
    VoiceSessionClosedError,
    VoiceSessionNotFoundError,
    VoiceSessionService,
    VoiceSessionStatus,
)
from app.services.voiceprint_service import (
    FamilyVerificationResult,
    FamilyVoiceMatch,
)
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
from app.utils.audio_quality import (
    AudioQualityError,
    AudioQualityResult,
    analyze_standard_wav_quality,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice-sessions", tags=["voice-sessions"])


def get_family_repository(settings: Settings = Depends(get_settings)) -> FamilyRepository:
    return FamilyRepository(settings.database_path)


def get_voice_session_repository(
    settings: Settings = Depends(get_settings),
) -> VoiceSessionRepository:
    return VoiceSessionRepository(settings.database_path)


def get_voice_session_read_service(
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
    voice_session_repository: VoiceSessionRepository = Depends(get_voice_session_repository),
) -> VoiceSessionService:
    return VoiceSessionService(
        voice_session_repository=voice_session_repository,
        family_repository=family_repository,
        speaker_threshold=settings.speaker_threshold,
        anti_spoofing_threshold=settings.anti_spoofing_threshold,
        repeated_spoof_chunks=settings.voice_session_repeated_spoof_chunks,
        strong_spoof_score=settings.voice_session_strong_spoof_score,
        family_confirm_chunks=settings.voice_session_family_confirm_chunks,
    )


def _to_candidate_response(match: FamilyVoiceMatch) -> FamilyCandidateResponse:
    return FamilyCandidateResponse(
        family_id=match.family_id,
        name=match.name,
        relation=match.relation,
        similarity=match.similarity,
        sample_count=match.sample_count,
        max_similarity=match.max_similarity,
        mean_similarity=match.mean_similarity,
        median_similarity=match.median_similarity,
        weighted_mean_similarity=match.weighted_mean_similarity,
        weighted_median_similarity=match.weighted_median_similarity,
        profile_threshold=match.profile_threshold,
        confidence_score=match.confidence_score,
        sample_quality=match.sample_quality,
        low_quality_sample_count=match.low_quality_sample_count,
    )


def _family_result_to_response(result: FamilyVerificationResult) -> VerifyFamilyResponse:
    best_match = (
        _to_candidate_response(result.best_match)
        if result.best_match is not None
        else None
    )

    return VerifyFamilyResponse(
        is_registered_family=result.is_registered_family,
        best_match=best_match,
        threshold=result.threshold,
        candidates=[
            _to_candidate_response(candidate)
            for candidate in result.candidates
        ],
        message=result.message,
        model_name=result.model_name,
    )


def _anti_spoofing_result_to_response(result: AntiSpoofingResult) -> AntiSpoofingResponse:
    return AntiSpoofingResponse(
        analysis_status=(
            "additional_confirmation"
            if result.message == "additional_confirmation"
            else "complete"
        ),
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
    )


def _status_to_response(result: VoiceSessionStatus) -> VoiceSessionStatusResponse:
    best_family_match = (
        _to_candidate_response(result.best_family_match)
        if result.best_family_match is not None
        else None
    )

    return VoiceSessionStatusResponse(
        session_id=result.session_id,
        status=result.status,
        created_at=result.created_at,
        updated_at=result.updated_at,
        ended_at=result.ended_at,
        chunks_analyzed=result.chunks_analyzed,
        total_chunks=result.total_chunks,
        analyzable_chunks=result.analyzable_chunks,
        skipped_chunks=result.skipped_chunks,
        is_spoofed=result.is_spoofed,
        is_registered_family=result.is_registered_family,
        risk_level=result.risk_level,
        message=result.message,
        rolling_family_confidence=result.rolling_family_confidence,
        rolling_mismatch_confidence=result.rolling_mismatch_confidence,
        trusted_chunks=result.trusted_chunks,
        mismatch_chunks=result.mismatch_chunks,
        max_spoof_score=result.max_spoof_score,
        max_spoof_chunk_index=result.max_spoof_chunk_index,
        suspicious_chunks=result.suspicious_chunks,
        required_spoof_chunks=result.required_spoof_chunks,
        strong_spoof_score=result.strong_spoof_score,
        best_family_match=best_family_match,
        family_match_chunks=result.family_match_chunks,
        required_family_match_chunks=result.required_family_match_chunks,
        speaker_threshold=result.speaker_threshold,
        anti_spoofing_threshold=result.anti_spoofing_threshold,
    )


def _quality_to_response(result: AudioQualityResult) -> AudioQualityResponse:
    return AudioQualityResponse(
        is_analyzable=result.is_analyzable,
        message=result.message,
        duration_seconds=result.duration_seconds,
        rms_energy=result.rms_energy,
        peak_amplitude=result.peak_amplitude,
        speech_ratio=result.speech_ratio,
        estimated_snr_db=result.estimated_snr_db,
    )


def _chunk_analysis_to_response(
    result: VoiceSessionChunkAnalysis,
) -> VoiceSessionChunkResponse:
    return VoiceSessionChunkResponse(
        session_id=result.session_id,
        chunk_index=result.chunk_index,
        is_analyzable=result.is_analyzable,
        quality=_quality_to_response(result.quality),
        is_trusted_chunk=result.is_trusted_chunk,
        final_decision=result.final_decision,
        family_verification=(
            _family_result_to_response(result.family_verification)
            if result.family_verification is not None
            else None
        ),
        anti_spoofing=(
            _anti_spoofing_result_to_response(result.anti_spoofing)
            if result.anti_spoofing is not None
            else None
        ),
        rolling_result=_status_to_response(result.rolling_result),
    )


def _raise_audio_http_error(exc: Exception) -> None:
    if isinstance(exc, MissingAudioFileError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, UnsupportedAudioFormatError):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, UploadFileTooLargeError):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    if isinstance(exc, AudioTooShortError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, AudioDecodingError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, AudioValidationError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    raise exc


@router.post("/start", response_model=VoiceSessionStartResponse)
async def start_voice_session(
    service: VoiceSessionService = Depends(get_voice_session_read_service),
) -> VoiceSessionStartResponse:
    """Start a chunk-based voice analysis session."""

    try:
        return VoiceSessionStartResponse(**_status_to_response(service.start_session()).model_dump())
    except Exception as exc:
        logger.exception("Unexpected error while starting voice session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while starting voice session",
        ) from exc


@router.post("/{session_id}/chunks", response_model=VoiceSessionChunkResponse)
async def analyze_voice_session_chunk(
    session_id: str,
    chunk_index: int | None = Form(
        default=None,
        description="Optional zero-based chunk index. If omitted, the server assigns the next index.",
    ),
    audio_file: UploadFile | None = File(
        default=None,
        description="Short audio chunk. Recommended length: 3 to 5 seconds.",
    ),
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
    voice_session_repository: VoiceSessionRepository = Depends(get_voice_session_repository),
) -> VoiceSessionChunkResponse:
    """Analyze one uploaded audio chunk and return the updated rolling result."""

    temp_paths: list[Path | None] = []

    try:
        session = voice_session_repository.get(session_id)
        if session is None:
            raise VoiceSessionNotFoundError("voice session not found")
        if session.status != "active":
            raise VoiceSessionClosedError("voice session is already ended")

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

        audio_quality = await run_in_threadpool(
            analyze_standard_wav_quality,
            wav_path=wav_file,
            target_sample_rate=settings.target_sample_rate,
            min_analyzable_seconds=settings.voice_session_min_analyzable_seconds,
            min_rms_energy=settings.voice_session_min_rms_energy,
            min_speech_ratio=settings.voice_session_min_speech_ratio,
            min_estimated_snr_db=settings.voice_session_min_estimated_snr_db,
        )

        speaker_service = get_speaker_service() if audio_quality.is_analyzable else None
        anti_spoofing_service = (
            get_anti_spoofing_service()
            if audio_quality.is_analyzable
            else None
        )
        service = VoiceSessionService(
            voice_session_repository=voice_session_repository,
            family_repository=family_repository,
            speaker_threshold=settings.speaker_threshold,
            anti_spoofing_threshold=(
                anti_spoofing_service.threshold
                if anti_spoofing_service is not None
                else settings.anti_spoofing_threshold
            ),
            repeated_spoof_chunks=settings.voice_session_repeated_spoof_chunks,
            strong_spoof_score=settings.voice_session_strong_spoof_score,
            family_confirm_chunks=settings.voice_session_family_confirm_chunks,
            speaker_service=speaker_service,
            anti_spoofing_service=anti_spoofing_service,
        )
        result = await run_in_threadpool(
            service.analyze_chunk,
            session_id=session_id,
            wav_path=wav_file,
            audio_quality=audio_quality,
            chunk_index=chunk_index,
        )
        return _chunk_analysis_to_response(result)

    except (
        MissingAudioFileError,
        UnsupportedAudioFormatError,
        UploadFileTooLargeError,
        AudioTooShortError,
        AudioDecodingError,
        AudioValidationError,
    ) as exc:
        _raise_audio_http_error(exc)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except VoiceSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except VoiceSessionClosedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except AudioQualityError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (SpeakerVerificationError, AntiSpoofingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while analyzing voice session chunk")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while analyzing voice session chunk",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)


@router.get("/{session_id}", response_model=VoiceSessionStatusResponse)
async def get_voice_session_status(
    session_id: str,
    service: VoiceSessionService = Depends(get_voice_session_read_service),
) -> VoiceSessionStatusResponse:
    """Read the rolling result for a chunk-based voice analysis session."""

    try:
        return _status_to_response(service.get_status(session_id))
    except VoiceSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while reading voice session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while reading voice session",
        ) from exc


@router.post("/{session_id}/end", response_model=VoiceSessionStatusResponse)
async def end_voice_session(
    session_id: str,
    service: VoiceSessionService = Depends(get_voice_session_read_service),
) -> VoiceSessionStatusResponse:
    """End a chunk-based voice analysis session."""

    try:
        return _status_to_response(service.end_session(session_id))
    except VoiceSessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while ending voice session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error while ending voice session",
        ) from exc
