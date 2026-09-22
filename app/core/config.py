import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Values can be overridden with environment variables or a local .env file.
    Example:
        ISFAM_SPEAKER_THRESHOLD=0.65
        ISFAM_MAX_UPLOAD_SIZE_MB=20
    """

    app_name: str = "IsFAM Speaker Verification API"
    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: Tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "capacitor://localhost",
        "ionic://localhost",
    )
    cors_allowed_origin_regex: str | None = (
        r"^https?://("
        r"localhost|"
        r"127\.0\.0\.1|"
        r"10\.0\.2\.2|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    )

    # SpeechBrain ECAPA-TDNN model published on Hugging Face.
    speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_model_dir: Path = Path("pretrained_models/spkrec-ecapa-voxceleb")
    database_path: Path = Path("data/isfam.sqlite3")
    demo_sample_dir: Path = Path("demo_samples")

    # Postgres connection for the product domains (auth, family, devices, ...).
    # Existing SQLite-backed domains keep using `database_path` until they migrate.
    database_url: str = "postgresql+psycopg2://isfam:isfam@127.0.0.1:5433/isfam"

    # Auth / JWT settings.
    jwt_secret_key: str = "dev-only-change-me"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 14
    # Demo policy: verification_code is fixed instead of sending a real SMS OTP.
    auth_fixed_verification_code: str = "123456"

    # Select the fast local spectral model by default; keep Transformers as fallback.
    anti_spoofing_backend: str = "spectral"
    anti_spoofing_model_name: str = "Vansh180/deepfake-audio-wav2vec2"
    anti_spoofing_model_version: str = "2026-09-spectral-v1"
    anti_spoofing_model_dir: Path = Path("pretrained_models/deepfake-audio-wav2vec2")
    spectral_anti_spoofing_model_name: str = "isfam/spectral-mlp-dfadd-v1"
    spectral_anti_spoofing_model_path: Path = Path(
        "app/assets/spectral_anti_spoof_v1.npz"
    )
    anti_spoofing_threshold: float = 0.5107068233191967
    anti_spoofing_spoof_labels: Tuple[str, ...] = (
        "spoof",
        "fake",
        "deepfake",
        "synthetic",
        "generated",
        "label_1",
    )
    anti_spoofing_max_audio_seconds: float = 60.0
    anti_spoofing_window_seconds: float = 5.0
    anti_spoofing_hop_seconds: float = 2.5
    anti_spoofing_batch_size: int = 4
    anti_spoofing_max_concurrency: int = 2
    preload_models: bool = True
    preload_speaker_model: bool = True

    # Chunk session policy. These values reduce noisy decisions without model retraining.
    voice_session_min_analyzable_seconds: float = 2.0
    voice_session_min_rms_energy: float = 0.005
    voice_session_min_speech_ratio: float = 0.25
    voice_session_repeated_spoof_chunks: int = 2
    voice_session_strong_spoof_score: float = 0.80
    voice_session_family_confirm_chunks: int = 2

    # Tuned from current family_real pairwise evaluation.
    speaker_threshold: float = 0.65

    # CPU is the safest default. Set ISFAM_DEVICE=cuda on a CUDA machine.
    device: str = "cpu"

    allowed_audio_extensions: Tuple[str, ...] = ("wav", "mp3", "m4a")
    max_upload_size_mb: int = 25
    min_audio_seconds: float = 1.0
    target_sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.anti_spoofing_backend not in {"spectral", "transformers"}:
            raise ValueError(
                "ISFAM_ANTI_SPOOFING_BACKEND must be spectral or transformers"
            )
        if not -1.0 <= self.speaker_threshold <= 1.0:
            raise ValueError("ISFAM_SPEAKER_THRESHOLD must be between -1.0 and 1.0")
        if not 0.0 <= self.anti_spoofing_threshold <= 1.0:
            raise ValueError("ISFAM_ANTI_SPOOFING_THRESHOLD must be between 0.0 and 1.0")
        if self.anti_spoofing_max_audio_seconds < 1.0:
            raise ValueError(
                "ISFAM_ANTI_SPOOFING_MAX_AUDIO_SECONDS must be greater than or equal to 1.0"
            )
        if self.anti_spoofing_window_seconds < 1.0:
            raise ValueError(
                "ISFAM_ANTI_SPOOFING_WINDOW_SECONDS must be greater than or equal to 1.0"
            )
        if self.anti_spoofing_hop_seconds <= 0:
            raise ValueError("ISFAM_ANTI_SPOOFING_HOP_SECONDS must be greater than 0")
        if self.anti_spoofing_batch_size < 1:
            raise ValueError("ISFAM_ANTI_SPOOFING_BATCH_SIZE must be at least 1")
        if self.anti_spoofing_max_concurrency < 1:
            raise ValueError("ISFAM_ANTI_SPOOFING_MAX_CONCURRENCY must be at least 1")
        if self.voice_session_min_analyzable_seconds < self.min_audio_seconds:
            raise ValueError(
                "ISFAM_VOICE_SESSION_MIN_ANALYZABLE_SECONDS must be greater than "
                "or equal to ISFAM_MIN_AUDIO_SECONDS"
            )
        if not 0.0 <= self.voice_session_min_rms_energy <= 1.0:
            raise ValueError(
                "ISFAM_VOICE_SESSION_MIN_RMS_ENERGY must be between 0.0 and 1.0"
            )
        if not 0.0 <= self.voice_session_min_speech_ratio <= 1.0:
            raise ValueError(
                "ISFAM_VOICE_SESSION_MIN_SPEECH_RATIO must be between 0.0 and 1.0"
            )
        if self.voice_session_repeated_spoof_chunks < 1:
            raise ValueError(
                "ISFAM_VOICE_SESSION_REPEATED_SPOOF_CHUNKS must be greater than or equal to 1"
            )
        if not 0.0 <= self.voice_session_strong_spoof_score <= 1.0:
            raise ValueError(
                "ISFAM_VOICE_SESSION_STRONG_SPOOF_SCORE must be between 0.0 and 1.0"
            )
        if self.voice_session_family_confirm_chunks < 1:
            raise ValueError(
                "ISFAM_VOICE_SESSION_FAMILY_CONFIRM_CHUNKS must be greater than or equal to 1"
            )
        if self.max_upload_size_mb < 1:
            raise ValueError("ISFAM_MAX_UPLOAD_SIZE_MB must be greater than or equal to 1")
        if self.min_audio_seconds < 0.1:
            raise ValueError("ISFAM_MIN_AUDIO_SECONDS must be greater than or equal to 0.1")
        if self.target_sample_rate < 8000:
            raise ValueError("ISFAM_TARGET_SAMPLE_RATE must be greater than or equal to 8000")
        if not self.database_url:
            raise ValueError("ISFAM_DATABASE_URL must not be empty")
        if not self.jwt_secret_key:
            raise ValueError("ISFAM_JWT_SECRET_KEY must not be empty")
        if self.jwt_access_token_expire_minutes < 1:
            raise ValueError(
                "ISFAM_JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be greater than or equal to 1"
            )
        if self.jwt_refresh_token_expire_days < 1:
            raise ValueError(
                "ISFAM_JWT_REFRESH_TOKEN_EXPIRE_DAYS must be greater than or equal to 1"
            )
        if not self.auth_fixed_verification_code:
            raise ValueError("ISFAM_AUTH_FIXED_VERIFICATION_CODE must not be empty")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def _load_dotenv(dotenv_path: Path = Path(".env")) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without adding extra dependencies."""

    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def _get_env(name: str, default: str, dotenv_values: dict[str, str]) -> str:
    """Read ISFAM_* setting from environment first, then .env, then default."""

    return os.getenv(name) or dotenv_values.get(name, default)


def _get_int_env(name: str, default: int, dotenv_values: dict[str, str]) -> int:
    return int(_get_env(name, str(default), dotenv_values))


def _get_float_env(name: str, default: float, dotenv_values: dict[str, str]) -> float:
    return float(_get_env(name, str(default), dotenv_values))


def _get_bool_env(name: str, default: bool, dotenv_values: dict[str, str]) -> bool:
    raw_value = _get_env(name, str(default), dotenv_values).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _get_tuple_env(
    name: str,
    default: Tuple[str, ...],
    dotenv_values: dict[str, str],
) -> Tuple[str, ...]:
    raw_value = _get_env(name, ",".join(default), dotenv_values)
    return tuple(item.strip().lower().lstrip(".") for item in raw_value.split(",") if item.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so every module uses the same config instance."""

    dotenv_values = _load_dotenv()

    return Settings(
        app_name=_get_env("ISFAM_APP_NAME", "IsFAM Speaker Verification API", dotenv_values),
        api_v1_prefix=_get_env("ISFAM_API_V1_PREFIX", "/api/v1", dotenv_values),
        cors_allowed_origins=_get_tuple_env(
            "ISFAM_CORS_ALLOWED_ORIGINS",
            (
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "http://127.0.0.1:5174",
                "capacitor://localhost",
                "ionic://localhost",
            ),
            dotenv_values,
        ),
        cors_allowed_origin_regex=(
            _get_env(
                "ISFAM_CORS_ALLOWED_ORIGIN_REGEX",
                (
                    r"^https?://("
                    r"localhost|"
                    r"127\.0\.0\.1|"
                    r"10\.0\.2\.2|"
                    r"192\.168\.\d{1,3}\.\d{1,3}|"
                    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
                    r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
                    r")(:\d+)?$"
                ),
                dotenv_values,
            )
            or None
        ),
        speaker_model_name=_get_env(
            "ISFAM_SPEAKER_MODEL_NAME",
            "speechbrain/spkrec-ecapa-voxceleb",
            dotenv_values,
        ),
        speaker_model_dir=Path(
            _get_env(
                "ISFAM_SPEAKER_MODEL_DIR",
                "pretrained_models/spkrec-ecapa-voxceleb",
                dotenv_values,
            )
        ),
        database_path=Path(
            _get_env(
                "ISFAM_DATABASE_PATH",
                "data/isfam.sqlite3",
                dotenv_values,
            )
        ),
        demo_sample_dir=Path(
            _get_env(
                "ISFAM_DEMO_SAMPLE_DIR",
                "demo_samples",
                dotenv_values,
            )
        ),
        anti_spoofing_backend=_get_env(
            "ISFAM_ANTI_SPOOFING_BACKEND", "spectral", dotenv_values
        ).strip().lower(),
        anti_spoofing_model_name=_get_env(
            "ISFAM_ANTI_SPOOFING_MODEL_NAME",
            "Vansh180/deepfake-audio-wav2vec2",
            dotenv_values,
        ),
        anti_spoofing_model_version=_get_env(
            "ISFAM_ANTI_SPOOFING_MODEL_VERSION",
            "2026-09-spectral-v1",
            dotenv_values,
        ),
        anti_spoofing_model_dir=Path(
            _get_env(
                "ISFAM_ANTI_SPOOFING_MODEL_DIR",
                "pretrained_models/deepfake-audio-wav2vec2",
                dotenv_values,
            )
        ),
        spectral_anti_spoofing_model_name=_get_env(
            "ISFAM_SPECTRAL_ANTI_SPOOFING_MODEL_NAME",
            "isfam/spectral-mlp-dfadd-v1",
            dotenv_values,
        ),
        spectral_anti_spoofing_model_path=Path(
            _get_env(
                "ISFAM_SPECTRAL_ANTI_SPOOFING_MODEL_PATH",
                "app/assets/spectral_anti_spoof_v1.npz",
                dotenv_values,
            )
        ),
        anti_spoofing_threshold=_get_float_env(
            "ISFAM_ANTI_SPOOFING_THRESHOLD",
            0.5107068233191967,
            dotenv_values,
        ),
        anti_spoofing_spoof_labels=_get_tuple_env(
            "ISFAM_ANTI_SPOOFING_SPOOF_LABELS",
            ("spoof", "fake", "deepfake", "synthetic", "generated", "label_1"),
            dotenv_values,
        ),
        anti_spoofing_max_audio_seconds=_get_float_env(
            "ISFAM_ANTI_SPOOFING_MAX_AUDIO_SECONDS",
            60.0,
            dotenv_values,
        ),
        anti_spoofing_window_seconds=_get_float_env(
            "ISFAM_ANTI_SPOOFING_WINDOW_SECONDS",
            5.0,
            dotenv_values,
        ),
        anti_spoofing_hop_seconds=_get_float_env(
            "ISFAM_ANTI_SPOOFING_HOP_SECONDS",
            2.5,
            dotenv_values,
        ),
        anti_spoofing_batch_size=_get_int_env(
            "ISFAM_ANTI_SPOOFING_BATCH_SIZE", 4, dotenv_values
        ),
        anti_spoofing_max_concurrency=_get_int_env(
            "ISFAM_ANTI_SPOOFING_MAX_CONCURRENCY", 2, dotenv_values
        ),
        preload_models=_get_bool_env("ISFAM_PRELOAD_MODELS", True, dotenv_values),
        preload_speaker_model=_get_bool_env(
            "ISFAM_PRELOAD_SPEAKER_MODEL", True, dotenv_values
        ),
        voice_session_min_analyzable_seconds=_get_float_env(
            "ISFAM_VOICE_SESSION_MIN_ANALYZABLE_SECONDS",
            2.0,
            dotenv_values,
        ),
        voice_session_min_rms_energy=_get_float_env(
            "ISFAM_VOICE_SESSION_MIN_RMS_ENERGY",
            0.005,
            dotenv_values,
        ),
        voice_session_min_speech_ratio=_get_float_env(
            "ISFAM_VOICE_SESSION_MIN_SPEECH_RATIO",
            0.25,
            dotenv_values,
        ),
        voice_session_repeated_spoof_chunks=_get_int_env(
            "ISFAM_VOICE_SESSION_REPEATED_SPOOF_CHUNKS",
            2,
            dotenv_values,
        ),
        voice_session_strong_spoof_score=_get_float_env(
            "ISFAM_VOICE_SESSION_STRONG_SPOOF_SCORE",
            0.80,
            dotenv_values,
        ),
        voice_session_family_confirm_chunks=_get_int_env(
            "ISFAM_VOICE_SESSION_FAMILY_CONFIRM_CHUNKS",
            2,
            dotenv_values,
        ),
        speaker_threshold=_get_float_env("ISFAM_SPEAKER_THRESHOLD", 0.65, dotenv_values),
        device=_get_env("ISFAM_DEVICE", "cpu", dotenv_values),
        allowed_audio_extensions=_get_tuple_env(
            "ISFAM_ALLOWED_AUDIO_EXTENSIONS",
            ("wav", "mp3", "m4a"),
            dotenv_values,
        ),
        max_upload_size_mb=_get_int_env("ISFAM_MAX_UPLOAD_SIZE_MB", 25, dotenv_values),
        min_audio_seconds=_get_float_env("ISFAM_MIN_AUDIO_SECONDS", 1.0, dotenv_values),
        target_sample_rate=_get_int_env("ISFAM_TARGET_SAMPLE_RATE", 16000, dotenv_values),
        database_url=_get_env(
            "ISFAM_DATABASE_URL",
            "postgresql+psycopg2://isfam:isfam@127.0.0.1:5433/isfam",
            dotenv_values,
        ),
        jwt_secret_key=_get_env("ISFAM_JWT_SECRET_KEY", "dev-only-change-me", dotenv_values),
        jwt_access_token_expire_minutes=_get_int_env(
            "ISFAM_JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30, dotenv_values
        ),
        jwt_refresh_token_expire_days=_get_int_env(
            "ISFAM_JWT_REFRESH_TOKEN_EXPIRE_DAYS", 14, dotenv_values
        ),
        auth_fixed_verification_code=_get_env(
            "ISFAM_AUTH_FIXED_VERIFICATION_CODE", "123456", dotenv_values
        ),
    )
