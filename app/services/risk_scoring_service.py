from dataclasses import dataclass

from app.services.anti_spoofing_service import AntiSpoofingResult
from app.services.voiceprint_service import FamilyVerificationResult


@dataclass(frozen=True)
class RiskScoringResult:
    is_trusted: bool
    risk_level: str
    risk_score: float
    family_confidence: float
    mismatch_confidence: float
    final_decision: str
    reasons: list[str]


class RiskScoringService:
    """Family-first risk scoring for voice phishing detection.

    IsFAM treats "not similar to registered family" as the primary danger
    signal. Anti-spoofing is still useful, but it is a supporting signal because
    a real human impersonator is also unsafe in this service domain.
    """

    def __init__(self, strong_spoof_score: float):
        self.strong_spoof_score = strong_spoof_score

    def score_secure_voice(
        self,
        family_result: FamilyVerificationResult,
        anti_spoofing_result: AntiSpoofingResult,
    ) -> RiskScoringResult:
        family_score, family_reasons = self._score_family_match(family_result)
        spoof_score, spoof_reasons = self._score_spoof_signal(anti_spoofing_result)
        stability_score, stability_reasons = self._score_voiceprint_stability(family_result)

        risk_score = self._apply_family_first_policy(
            base_score=min(1.0, family_score * 0.80 + spoof_score * 0.15 + stability_score * 0.05),
            family_result=family_result,
            anti_spoofing_result=anti_spoofing_result,
        )
        risk_level = self._risk_level(risk_score)
        family_confidence = self._family_confidence(family_result)
        mismatch_confidence = round(max(risk_score, 1.0 - family_confidence), 4)
        is_trusted = (
            risk_level == "safe"
            and family_result.is_registered_family
            and not anti_spoofing_result.is_spoofed
        )

        return RiskScoringResult(
            is_trusted=is_trusted,
            risk_level=risk_level,
            risk_score=risk_score,
            family_confidence=family_confidence,
            mismatch_confidence=mismatch_confidence,
            final_decision=self._final_decision(
                is_trusted=is_trusted,
                family_result=family_result,
                anti_spoofing_result=anti_spoofing_result,
                risk_level=risk_level,
            ),
            reasons=family_reasons + spoof_reasons + stability_reasons,
        )

    @staticmethod
    def _family_confidence(family_result: FamilyVerificationResult) -> float:
        best_match = family_result.best_match
        if best_match is None:
            return 0.0

        threshold = family_result.threshold
        margin = best_match.similarity - threshold
        if margin >= 0:
            confidence = 0.75 + min(0.25, margin / 0.20 * 0.25)
        else:
            confidence = max(0.0, 0.75 + margin / 0.20 * 0.75)

        if best_match.confidence_score is not None:
            confidence = confidence * (0.7 + best_match.confidence_score * 0.3)

        if best_match.low_quality_sample_count:
            confidence -= min(0.08, best_match.low_quality_sample_count * 0.02)

        return round(max(0.0, min(1.0, confidence)), 4)

    def _apply_family_first_policy(
        self,
        base_score: float,
        family_result: FamilyVerificationResult,
        anti_spoofing_result: AntiSpoofingResult,
    ) -> float:
        """Force family mismatch to remain visible even when spoof score is low."""

        best_match = family_result.best_match
        risk_score = base_score

        if best_match is None:
            return 1.0

        margin = best_match.similarity - family_result.threshold
        if not family_result.is_registered_family:
            if margin < -0.05:
                risk_score = max(risk_score, 0.75)
            else:
                risk_score = max(risk_score, 0.45)

        if family_result.is_registered_family and margin < 0.03:
            # A barely passing match may be a degraded family recording or a
            # similar-sounding impostor. Require another sample instead of trust.
            risk_score = max(risk_score, 0.35)

        if family_result.is_registered_family and anti_spoofing_result.is_spoofed:
            risk_score = max(risk_score, 0.55)

        if (
            family_result.is_registered_family
            and anti_spoofing_result.spoof_score >= anti_spoofing_result.threshold * 0.7
        ):
            # The user-facing policy favors another confirmation over silently
            # trusting a family-like voice when the spoof score is borderline.
            risk_score = max(risk_score, 0.35)

        if anti_spoofing_result.spoof_score >= self.strong_spoof_score:
            risk_score = max(risk_score, 0.85)

        return round(min(1.0, risk_score), 4)

    @staticmethod
    def _score_family_match(result: FamilyVerificationResult) -> tuple[float, list[str]]:
        if result.best_match is None:
            return 1.0, ["등록된 가족 voiceprint가 없어 가족 여부를 확인할 수 없습니다."]

        similarity = result.best_match.similarity
        margin = round(similarity - result.threshold, 4)
        if result.is_registered_family:
            score = 0.0 if margin >= 0.08 else 0.25
            return score, [
                f"{result.best_match.name} voiceprint와 유사도 {similarity:.4f}로 기준보다 {margin:.4f} 높습니다."
            ]

        if margin >= -0.08:
            return 0.65, [
                f"가장 가까운 가족 voiceprint 유사도 {similarity:.4f}가 기준에 근접하지만 통과하지 못했습니다."
            ]

        return 1.0, [
            f"가장 가까운 가족 voiceprint 유사도 {similarity:.4f}가 기준보다 낮아 가족으로 신뢰하기 어렵습니다."
        ]

    def _score_spoof_signal(self, result: AntiSpoofingResult) -> tuple[float, list[str]]:
        if result.spoof_score >= self.strong_spoof_score:
            return 1.0, [
                f"AI 합성 음성 의심 점수 {result.spoof_score:.4f}가 강한 경고 기준에 도달했습니다."
            ]

        if result.is_spoofed:
            return 0.75, [
                f"AI 합성 음성 의심 점수 {result.spoof_score:.4f}가 탐지 기준을 넘었습니다."
            ]

        if result.spoof_score >= result.threshold * 0.7:
            return 0.35, [
                f"AI 합성 음성 의심 점수 {result.spoof_score:.4f}가 기준에 근접합니다."
            ]

        return 0.0, [
            f"AI 합성 음성 의심 점수 {result.spoof_score:.4f}가 기준보다 낮습니다."
        ]

    @staticmethod
    def _score_voiceprint_stability(
        result: FamilyVerificationResult,
    ) -> tuple[float, list[str]]:
        best_match = result.best_match
        if best_match is None:
            return 0.0, []

        if best_match.sample_count < 3:
            return 0.4, [
                f"{best_match.name} 등록 샘플이 {best_match.sample_count}개라 안정성 판단에는 추가 샘플이 필요합니다."
            ]

        if best_match.low_quality_sample_count > 0:
            return 0.2, [
                f"{best_match.name} 등록 샘플 중 {best_match.low_quality_sample_count}개는 품질 재검토 권장 대상입니다."
            ]

        if best_match.confidence_score is not None and best_match.confidence_score >= 0.8:
            return 0.0, [
                f"{best_match.sample_count}개 등록 샘플 기준 가족 프로필 분리 신뢰도가 높습니다."
            ]

        return 0.1, [
            f"{best_match.sample_count}개 등록 샘플 기준 가족 프로필 분리 신뢰도 확인이 필요합니다."
        ]

    @staticmethod
    def _risk_level(risk_score: float) -> str:
        if risk_score >= 0.65:
            return "danger"
        if risk_score >= 0.35:
            return "caution"
        return "safe"

    @staticmethod
    def _final_decision(
        is_trusted: bool,
        family_result: FamilyVerificationResult,
        anti_spoofing_result: AntiSpoofingResult,
        risk_level: str,
    ) -> str:
        if is_trusted:
            return "trusted_family_voice"
        if family_result.is_registered_family and anti_spoofing_result.is_spoofed:
            return "spoofed_family_like_voice"
        if not family_result.is_registered_family and anti_spoofing_result.is_spoofed:
            return "spoofed_unknown_voice"
        if not family_result.is_registered_family and risk_level == "danger":
            return "unregistered_voice_detected"
        if risk_level == "caution":
            return "family_voice_needs_confirmation"
        return "unknown_real_voice"
