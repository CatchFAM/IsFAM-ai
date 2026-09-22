from unittest import TestCase

from app.services.voice_session_service import VoiceSessionService


class VoiceSessionPolicyTest(TestCase):
    def test_borderline_chunk_is_not_marked_trusted(self):
        is_trusted, decision = VoiceSessionService._make_secure_decision(
            is_registered_family=True,
            is_spoofed=False,
            has_spoof_warning=True,
        )

        self.assertFalse(is_trusted)
        self.assertEqual(decision, "family_voice_needs_confirmation")

    def test_borderline_spoof_warning_overrides_family_trust(self):
        risk_level, message = VoiceSessionService._make_rolling_decision(
            is_registered_family=True,
            is_spoofed=False,
            has_spoof_warning=True,
            has_family_warning=True,
            rolling_mismatch_confidence=0.3,
            trusted_chunks=2,
            mismatch_chunks=0,
        )

        self.assertEqual(risk_level, "medium")
        self.assertEqual(message, "spoof_warning_needs_more_chunks")

    def test_clear_registered_family_remains_low_risk(self):
        risk_level, message = VoiceSessionService._make_rolling_decision(
            is_registered_family=True,
            is_spoofed=False,
            has_spoof_warning=False,
            has_family_warning=True,
            rolling_mismatch_confidence=0.2,
            trusted_chunks=2,
            mismatch_chunks=0,
        )

        self.assertEqual(risk_level, "low")
        self.assertEqual(message, "registered_family_likely")
