from unittest import TestCase

from app.api.routes.anti_spoofing import anti_spoofing_result_to_response
from app.schemas.anti_spoofing import AntiSpoofingAudioQuality
from app.services.anti_spoofing_service import AntiSpoofingResult, LabelScore


def _result(message: str = "additional_confirmation") -> AntiSpoofingResult:
    return AntiSpoofingResult(
        is_spoofed=False,
        spoof_score=0.4,
        threshold=0.5,
        predicted_label="uncertain",
        predicted_score=0.4,
        message=message,
        model_name="test-model",
        analyzed_segments=1,
        max_spoof_segment_index=0,
        segment_seconds=5.0,
        label_scores=[LabelScore(label="fake", score=0.4)],
    )


def _quality(is_analyzable: bool = True) -> AntiSpoofingAudioQuality:
    return AntiSpoofingAudioQuality(
        is_analyzable=is_analyzable,
        message="analyzable" if is_analyzable else "low_signal_to_noise",
        duration_seconds=5.0,
        rms_energy=0.05,
        peak_amplitude=0.5,
        speech_ratio=0.8,
        estimated_snr_db=30.0 if is_analyzable else 12.0,
    )


class AntiSpoofingResponseTest(TestCase):
    def test_borderline_result_requests_additional_confirmation(self):
        response = anti_spoofing_result_to_response(
            _result(),
            processing_time_ms=3.0,
            audio_quality=_quality(),
        )

        self.assertEqual(response.analysis_status, "additional_confirmation")

    def test_low_quality_takes_priority_over_model_warning(self):
        response = anti_spoofing_result_to_response(
            _result(),
            processing_time_ms=3.0,
            audio_quality=_quality(is_analyzable=False),
        )

        self.assertEqual(response.analysis_status, "more_voice_required")
