from pydantic import BaseModel, Field

from app.schemas.anti_spoofing import AntiSpoofingResponse
from app.schemas.voice import FamilyCandidateResponse, VerifyFamilyResponse


class VoiceSessionStatusResponse(BaseModel):
    """Rolling status for one chunk-based voice analysis session."""

    session_id: str = Field(..., examples=["9d4a5f2f4d2b4ab6bb5a7b957b04e51d"])
    status: str = Field(..., examples=["active"])
    created_at: str
    updated_at: str
    ended_at: str | None = None
    chunks_analyzed: int = Field(..., examples=[4])
    total_chunks: int = Field(
        ...,
        description="Uploaded chunks including low-quality chunks that were skipped.",
        examples=[5],
    )
    analyzable_chunks: int = Field(
        ...,
        description="Chunks that passed audio quality checks and were analyzed by AI models.",
        examples=[4],
    )
    skipped_chunks: int = Field(
        ...,
        description="Chunks skipped because they were too short, silent, or speech-poor.",
        examples=[1],
    )
    is_spoofed: bool = Field(
        ...,
        description="True when spoof evidence is strong or repeated across chunks.",
        examples=[False],
    )
    is_registered_family: bool = Field(
        ...,
        description="True when the same family member is confirmed across enough chunks.",
        examples=[True],
    )
    risk_level: str = Field(..., examples=["low"])
    message: str = Field(..., examples=["registered_family_likely"])
    rolling_family_confidence: float = Field(
        ...,
        description="Weighted confidence accumulated from analyzed chunks that the caller matches registered family.",
        examples=[0.82],
    )
    rolling_mismatch_confidence: float = Field(
        ...,
        description="Weighted confidence accumulated from analyzed chunks that the caller should not be trusted as family.",
        examples=[0.18],
    )
    trusted_chunks: int = Field(
        ...,
        description="Analyzable chunks whose family confidence was strong and spoof confidence was low.",
        examples=[3],
    )
    mismatch_chunks: int = Field(
        ...,
        description="Analyzable chunks whose mismatch confidence was high.",
        examples=[0],
    )
    max_spoof_score: float = Field(..., examples=[0.12])
    max_spoof_chunk_index: int | None = Field(default=None, examples=[2])
    suspicious_chunks: int = Field(
        ...,
        description="Analyzable chunks whose spoof score reached the anti-spoofing threshold.",
        examples=[1],
    )
    required_spoof_chunks: int = Field(
        ...,
        description="Number of suspicious chunks required before rolling result becomes spoofed.",
        examples=[2],
    )
    strong_spoof_score: float = Field(
        ...,
        description="Spoof score high enough to mark rolling result as spoofed immediately.",
        examples=[0.8],
    )
    best_family_match: FamilyCandidateResponse | None = None
    family_match_chunks: int = Field(
        ...,
        description="Number of chunks where best_family_match reached the speaker threshold.",
        examples=[2],
    )
    required_family_match_chunks: int = Field(
        ...,
        description="Number of matching chunks required before rolling result confirms family.",
        examples=[2],
    )
    speaker_threshold: float = Field(..., examples=[0.65])
    anti_spoofing_threshold: float = Field(..., examples=[0.5])


class VoiceSessionStartResponse(VoiceSessionStatusResponse):
    """Response returned after starting a new voice analysis session."""


class AudioQualityResponse(BaseModel):
    """Basic audio quality gate result for one uploaded chunk."""

    is_analyzable: bool = Field(..., examples=[True])
    message: str = Field(..., examples=["analyzable"])
    duration_seconds: float = Field(..., examples=[4.82])
    rms_energy: float = Field(..., examples=[0.0321])
    peak_amplitude: float = Field(..., examples=[0.41])
    speech_ratio: float = Field(..., examples=[0.76])
    estimated_snr_db: float = Field(..., examples=[28.4])


class VoiceSessionChunkResponse(BaseModel):
    """Result returned after analyzing one uploaded audio chunk."""

    session_id: str = Field(..., examples=["9d4a5f2f4d2b4ab6bb5a7b957b04e51d"])
    chunk_index: int = Field(..., examples=[0])
    is_analyzable: bool = Field(
        ...,
        description="False when the chunk was skipped before model inference.",
        examples=[True],
    )
    quality: AudioQualityResponse
    is_trusted_chunk: bool = Field(
        ...,
        description="True only when this chunk matches a registered family member and is not spoofed.",
        examples=[True],
    )
    final_decision: str = Field(..., examples=["trusted_family_voice"])
    family_verification: VerifyFamilyResponse | None
    anti_spoofing: AntiSpoofingResponse | None
    rolling_result: VoiceSessionStatusResponse
