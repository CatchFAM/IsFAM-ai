from pathlib import Path
from unittest import TestCase

from app.services.anti_spoofing_service import AntiSpoofingResult, LabelScore
from app.services.risk_scoring_service import RiskScoringService
from app.services.secure_voice_service import SecureVoiceVerificationService
from app.services.voiceprint_service import FamilyVerificationResult, FamilyVoiceMatch
from app.utils.audio_quality import AudioQualityResult


class _VoiceprintStub:
    def __init__(self, result: FamilyVerificationResult):
        self.result = result
        self.calls = 0

    def verify_family_voice(self, _: Path) -> FamilyVerificationResult:
        self.calls += 1
        return self.result


class _AntiSpoofingStub:
    def __init__(self, result: AntiSpoofingResult):
        self.result = result
        self.calls = 0

    def detect_file(self, _: Path) -> AntiSpoofingResult:
        self.calls += 1
        return self.result


def _family_result(
    is_family: bool,
    similarity: float | None = None,
) -> FamilyVerificationResult:
    similarity = similarity if similarity is not None else (0.82 if is_family else 0.40)
    match = FamilyVoiceMatch(
        family_id=1,
        name="엄마",
        relation="mother",
        similarity=similarity,
        sample_count=3,
        confidence_score=0.9,
    )
    return FamilyVerificationResult(
        is_registered_family=is_family,
        best_match=match,
        threshold=0.65,
        candidates=[match],
        message="registered_family_matched" if is_family else "no_registered_family_match",
        model_name="speaker-test-model",
    )


def _spoof_result(is_spoofed: bool, score: float | None = None) -> AntiSpoofingResult:
    score = score if score is not None else (0.90 if is_spoofed else 0.10)
    return AntiSpoofingResult(
        is_spoofed=is_spoofed,
        spoof_score=score,
        threshold=0.5,
        predicted_label="spoof" if is_spoofed else "real",
        predicted_score=score if is_spoofed else 0.9,
        message="spoof" if is_spoofed else "bonafide",
        model_name="spoof-test-model",
        analyzed_segments=1,
        max_spoof_segment_index=0,
        segment_seconds=5.0,
        label_scores=[LabelScore(label="spoof", score=score)],
    )


class SecureVoiceVerificationServiceTest(TestCase):
    def _verify(self, is_family: bool, is_spoofed: bool):
        family_stub = _VoiceprintStub(_family_result(is_family))
        spoof_stub = _AntiSpoofingStub(_spoof_result(is_spoofed))
        service = SecureVoiceVerificationService(
            voiceprint_service=family_stub,  # type: ignore[arg-type]
            anti_spoofing_service=spoof_stub,  # type: ignore[arg-type]
            risk_scoring_service=RiskScoringService(strong_spoof_score=0.8),
        )
        result = service.verify(Path("unused.wav"))
        self.assertEqual(family_stub.calls, 1)
        self.assertEqual(spoof_stub.calls, 1)
        self.assertGreaterEqual(result.processing_time_ms, 0.0)
        return result

    def test_real_registered_family_is_trusted(self):
        result = self._verify(is_family=True, is_spoofed=False)
        self.assertTrue(result.risk.is_trusted)
        self.assertEqual(result.risk.final_decision, "trusted_family_voice")

    def test_spoofed_family_voice_is_not_trusted(self):
        result = self._verify(is_family=True, is_spoofed=True)
        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.final_decision, "spoofed_family_like_voice")

    def test_real_unknown_voice_is_not_trusted(self):
        result = self._verify(is_family=False, is_spoofed=False)
        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.final_decision, "unregistered_voice_detected")

    def test_spoofed_unknown_voice_is_not_trusted(self):
        result = self._verify(is_family=False, is_spoofed=True)
        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.final_decision, "spoofed_unknown_voice")

    def test_barely_passing_family_match_requires_confirmation(self):
        family_stub = _VoiceprintStub(_family_result(True, similarity=0.66))
        spoof_stub = _AntiSpoofingStub(_spoof_result(False))
        result = SecureVoiceVerificationService(
            voiceprint_service=family_stub,  # type: ignore[arg-type]
            anti_spoofing_service=spoof_stub,  # type: ignore[arg-type]
            risk_scoring_service=RiskScoringService(strong_spoof_score=0.8),
        ).verify(Path("unused.wav"))
        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.risk_level, "caution")
        self.assertEqual(result.risk.final_decision, "family_voice_needs_confirmation")

    def test_borderline_spoof_score_requires_confirmation(self):
        family_stub = _VoiceprintStub(_family_result(True))
        spoof_stub = _AntiSpoofingStub(_spoof_result(False, score=0.36))
        result = SecureVoiceVerificationService(
            voiceprint_service=family_stub,  # type: ignore[arg-type]
            anti_spoofing_service=spoof_stub,  # type: ignore[arg-type]
            risk_scoring_service=RiskScoringService(strong_spoof_score=0.8),
        ).verify(Path("unused.wav"))

        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.risk_level, "caution")
        self.assertEqual(result.risk.final_decision, "family_voice_needs_confirmation")

    def test_low_quality_audio_runs_models_but_requires_more_voice(self):
        family_stub = _VoiceprintStub(_family_result(True))
        spoof_stub = _AntiSpoofingStub(_spoof_result(False))
        service = SecureVoiceVerificationService(
            voiceprint_service=family_stub,  # type: ignore[arg-type]
            anti_spoofing_service=spoof_stub,  # type: ignore[arg-type]
            risk_scoring_service=RiskScoringService(strong_spoof_score=0.8),
        )
        result = service.verify(
            Path("unused.wav"),
            audio_quality=AudioQualityResult(
                is_analyzable=False,
                message="low_energy_or_silence",
                duration_seconds=3.0,
                rms_energy=0.001,
                peak_amplitude=0.01,
                speech_ratio=0.1,
            ),
        )
        self.assertEqual(family_stub.calls, 1)
        self.assertEqual(spoof_stub.calls, 1)
        self.assertFalse(result.risk.is_trusted)
        self.assertEqual(result.risk.risk_level, "caution")
        self.assertEqual(result.risk.final_decision, "more_voice_required")

    def test_low_quality_does_not_promote_unreliable_model_output_to_danger(self):
        service = SecureVoiceVerificationService(
            voiceprint_service=_VoiceprintStub(_family_result(False)),  # type: ignore[arg-type]
            anti_spoofing_service=_AntiSpoofingStub(_spoof_result(True)),  # type: ignore[arg-type]
            risk_scoring_service=RiskScoringService(strong_spoof_score=0.8),
        )
        result = service.verify(
            Path("unused.wav"),
            audio_quality=AudioQualityResult(False, "too_little_speech", 3.0, 0.01, 0.1, 0.1),
        )
        self.assertEqual(result.risk.risk_level, "caution")
        self.assertEqual(result.risk.final_decision, "more_voice_required")
        self.assertTrue(result.anti_spoofing.is_spoofed)
