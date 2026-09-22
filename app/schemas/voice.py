from pydantic import BaseModel, ConfigDict, Field

from app.schemas.anti_spoofing import AntiSpoofingResponse


class VoiceCompareResponse(BaseModel):
    """Response body for speaker verification."""

    model_config = ConfigDict(protected_namespaces=())

    similarity: float = Field(
        ...,
        description="Cosine similarity between two speaker embeddings.",
        examples=[0.83],
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used for same/different speaker.",
        examples=[0.65],
    )
    is_same_speaker: bool = Field(
        ...,
        description="True when similarity is greater than or equal to threshold.",
        examples=[True],
    )
    message: str = Field(
        ...,
        description="same_speaker or different_speaker.",
        examples=["same_speaker"],
    )
    model_name: str = Field(
        ...,
        description="Pretrained speaker verification model name.",
        examples=["speechbrain/spkrec-ecapa-voxceleb"],
    )


class FamilyCandidateResponse(BaseModel):
    """Similarity result for one registered family voiceprint."""

    family_id: int = Field(..., examples=[1])
    name: str = Field(..., examples=["엄마"])
    relation: str = Field(..., examples=["mother"])
    similarity: float = Field(..., examples=[0.86])
    sample_count: int = Field(
        default=1,
        description="Number of registered samples grouped into this family profile.",
        examples=[3],
    )
    max_similarity: float | None = Field(default=None, examples=[0.88])
    mean_similarity: float | None = Field(default=None, examples=[0.84])
    median_similarity: float | None = Field(default=None, examples=[0.85])
    weighted_mean_similarity: float | None = Field(default=None, examples=[0.83])
    weighted_median_similarity: float | None = Field(default=None, examples=[0.84])
    profile_threshold: float | None = Field(
        default=None,
        description="Personalized threshold calculated from this family profile's registered samples.",
        examples=[0.65],
    )
    confidence_score: float | None = Field(
        default=None,
        description="Profile separation confidence from 0.0 to 1.0.",
        examples=[0.92],
    )
    sample_quality: str | None = Field(default=None, examples=["ok"])
    low_quality_sample_count: int = Field(default=0, examples=[1])


class VerifyFamilyResponse(BaseModel):
    """Response for comparing one call voice against all registered family voices."""

    model_config = ConfigDict(protected_namespaces=())

    is_registered_family: bool = Field(
        ...,
        description="True when the best match similarity is greater than or equal to threshold.",
        examples=[True],
    )
    best_match: FamilyCandidateResponse | None = Field(
        default=None,
        description="Most similar registered family member. Null when no family member exists.",
    )
    threshold: float = Field(..., examples=[0.65])
    candidates: list[FamilyCandidateResponse]
    message: str = Field(
        ...,
        examples=["registered_family_matched"],
    )
    model_name: str = Field(
        ...,
        examples=["speechbrain/spkrec-ecapa-voxceleb"],
    )


class VoiceAudioQualityResponse(BaseModel):
    """Quality measurements for a one-shot verification upload."""

    is_analyzable: bool
    message: str
    duration_seconds: float
    rms_energy: float
    peak_amplitude: float
    speech_ratio: float
    estimated_snr_db: float


class SecureVoiceVerificationResponse(BaseModel):
    """Combined family verification and anti-spoofing result."""

    analysis_status: str = Field(
        ...,
        description=(
            "complete, additional_confirmation for caution results, or "
            "more_voice_required when call audio quality is weak."
        ),
        examples=["complete"],
    )
    is_trusted: bool = Field(
        ...,
        description="True only when the voice matches a registered family member and is not spoofed.",
        examples=[True],
    )
    risk_level: str = Field(..., examples=["safe"])
    risk_score: float = Field(
        ...,
        description="Explainable IsFAM risk score from 0.0 to 1.0.",
        examples=[0.18],
    )
    family_confidence: float = Field(
        ...,
        description="Confidence that the voice belongs to the best matched family profile.",
        examples=[0.87],
    )
    mismatch_confidence: float = Field(
        ...,
        description="Confidence that the voice should not be trusted as registered family.",
        examples=[0.13],
    )
    final_decision: str = Field(..., examples=["trusted_family_voice"])
    decision_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons used for the final decision.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total server-side model orchestration time in milliseconds.",
        examples=[842.31],
    )
    family_model_time_ms: float = Field(
        ...,
        description="Family voice verification time in milliseconds.",
        examples=[231.45],
    )
    anti_spoofing_model_time_ms: float = Field(
        ...,
        description="AI-generated voice detection time in milliseconds.",
        examples=[610.52],
    )
    audio_quality: VoiceAudioQualityResponse = Field(
        ...,
        description="Input quality measurements used to adjust result confidence.",
    )
    family_verification: VerifyFamilyResponse
    anti_spoofing: AntiSpoofingResponse
