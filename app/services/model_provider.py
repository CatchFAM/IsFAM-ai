from functools import lru_cache

from app.core.config import get_settings
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.speaker_service import SpeakerVerificationService
from app.services.spectral_anti_spoofing_service import SpectralAntiSpoofingService


@lru_cache(maxsize=1)
def get_speaker_service() -> SpeakerVerificationService:
    """Create one speaker model instance and reuse it across all API routes."""

    return SpeakerVerificationService(get_settings())


@lru_cache(maxsize=1)
def get_anti_spoofing_service() -> AntiSpoofingService | SpectralAntiSpoofingService:
    """Create one anti-spoofing model instance and reuse it across API routes."""

    settings = get_settings()
    if settings.anti_spoofing_backend == "spectral":
        return SpectralAntiSpoofingService(settings)
    return AntiSpoofingService(settings)


def preload_models(*, include_speaker: bool = True) -> None:
    """Load and warm requested models before the API begins serving traffic."""

    anti_spoofing_service = get_anti_spoofing_service()
    anti_spoofing_service.warm_up()
    if include_speaker:
        speaker_service = get_speaker_service()
        speaker_service.warm_up()
