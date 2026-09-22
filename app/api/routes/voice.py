import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.repositories.family_repository import FamilyRepository
from app.schemas.anti_spoofing import AntiSpoofingLabelScore, AntiSpoofingResponse
from app.schemas.voice import (
    FamilyCandidateResponse,
    SecureVoiceVerificationResponse,
    VerifyFamilyResponse,
    VoiceAudioQualityResponse,
    VoiceCompareResponse,
)
from app.services.anti_spoofing_service import AntiSpoofingError, AntiSpoofingResult
from app.services.model_provider import get_anti_spoofing_service, get_speaker_service
from app.services.risk_scoring_service import RiskScoringService
from app.services.secure_voice_service import SecureVoiceVerificationService
from app.services.speaker_service import SpeakerVerificationError
from app.services.voiceprint_service import (
    FamilyVerificationResult,
    FamilyVoiceMatch,
    VoiceprintService,
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
from app.utils.audio_quality import AudioQualityError, analyze_standard_wav_quality

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


def get_family_repository(settings: Settings = Depends(get_settings)) -> FamilyRepository:
    return FamilyRepository(settings.database_path)


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


@router.post("/compare", response_model=VoiceCompareResponse)
async def compare_voice(
    audio_file_1: UploadFile | None = File(
        default=None,
        description="First audio file. Supported: wav, mp3, m4a.",
    ),
    audio_file_2: UploadFile | None = File(
        default=None,
        description="Second audio file. Supported: wav, mp3, m4a.",
    ),
    settings: Settings = Depends(get_settings),
) -> VoiceCompareResponse:
    """Compare two uploaded voices with pretrained speaker embeddings."""

    temp_paths: list[Path | None] = []

    try:
        if audio_file_1 is None or audio_file_2 is None:
            raise MissingAudioFileError("audio_file_1 and audio_file_2 are required")

        original_1 = await save_upload_file_to_temp(
            upload_file=audio_file_1,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_1)

        original_2 = await save_upload_file_to_temp(
            upload_file=audio_file_2,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_2)

        wav_1 = convert_audio_to_standard_wav(
            input_path=original_1,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_1)

        wav_2 = convert_audio_to_standard_wav(
            input_path=original_2,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_2)

        # Load the pretrained model only after the request files are valid.
        # The first valid request may take longer because the model is downloaded.
        speaker_service = get_speaker_service()
        result = speaker_service.compare_files(wav_1, wav_2)

        return VoiceCompareResponse(
            similarity=result.similarity,
            threshold=result.threshold,
            is_same_speaker=result.is_same_speaker,
            message=result.message,
            model_name=result.model_name,
        )

    except MissingAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SpeakerVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during voice comparison")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during voice comparison",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)


@router.post("/verify-family", response_model=VerifyFamilyResponse)
async def verify_family_voice(
    audio_file: UploadFile | None = File(
        default=None,
        description="New call voice file to compare against all registered family voiceprints.",
    ),
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> VerifyFamilyResponse:
    """Compare one uploaded voice against every registered family voiceprint."""

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

        # Load the pretrained model only after upload validation and wav conversion.
        voiceprint_service = VoiceprintService(
            family_repository=family_repository,
            speaker_service=get_speaker_service(),
        )
        result = voiceprint_service.verify_family_voice(wav_file)
        return _family_result_to_response(result)

    except MissingAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SpeakerVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during family voice verification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during family voice verification",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)


@router.post("/verify", response_model=SecureVoiceVerificationResponse)
@router.post(
    "/verify-family-secure",
    response_model=SecureVoiceVerificationResponse,
    deprecated=True,
    include_in_schema=False,
)
async def verify_voice(
    audio_file: UploadFile | None = File(
        default=None,
        description="New call voice file for family verification plus anti-spoofing.",
    ),
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> SecureVoiceVerificationResponse:
    """Run family verification and AI-generated voice detection in one request."""

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

        audio_quality = await run_in_threadpool(
            analyze_standard_wav_quality,
            wav_path=wav_file,
            target_sample_rate=settings.target_sample_rate,
            min_analyzable_seconds=settings.voice_session_min_analyzable_seconds,
            min_rms_energy=settings.voice_session_min_rms_energy,
            min_speech_ratio=settings.voice_session_min_speech_ratio,
            min_estimated_snr_db=settings.voice_session_min_estimated_snr_db,
        )
        verification_service = SecureVoiceVerificationService(
            voiceprint_service=VoiceprintService(
                family_repository=family_repository,
                speaker_service=get_speaker_service(),
            ),
            anti_spoofing_service=get_anti_spoofing_service(),
            risk_scoring_service=RiskScoringService(
                strong_spoof_score=settings.voice_session_strong_spoof_score,
            ),
        )
        result = await run_in_threadpool(
            verification_service.verify,
            wav_file,
            audio_quality,
        )

        return SecureVoiceVerificationResponse(
            analysis_status=(
                "more_voice_required"
                if not audio_quality.is_analyzable
                else (
                    "additional_confirmation"
                    if result.risk.risk_level == "caution"
                    else "complete"
                )
            ),
            is_trusted=result.risk.is_trusted,
            risk_level=result.risk.risk_level,
            risk_score=result.risk.risk_score,
            family_confidence=result.risk.family_confidence,
            mismatch_confidence=result.risk.mismatch_confidence,
            final_decision=result.risk.final_decision,
            decision_reasons=result.risk.reasons,
            processing_time_ms=result.processing_time_ms,
            family_model_time_ms=result.family_model_time_ms,
            anti_spoofing_model_time_ms=result.anti_spoofing_model_time_ms,
            audio_quality=VoiceAudioQualityResponse(
                is_analyzable=audio_quality.is_analyzable,
                message=audio_quality.message,
                duration_seconds=audio_quality.duration_seconds,
                rms_energy=audio_quality.rms_energy,
                peak_amplitude=audio_quality.peak_amplitude,
                speech_ratio=audio_quality.speech_ratio,
                estimated_snr_db=audio_quality.estimated_snr_db,
            ),
            family_verification=_family_result_to_response(result.family_verification),
            anti_spoofing=_anti_spoofing_result_to_response(result.anti_spoofing),
        )

    except MissingAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
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
        logger.exception("Unexpected error during secure family voice verification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during secure family voice verification",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)
