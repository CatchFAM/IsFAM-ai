import math
from pathlib import Path
import tempfile
from unittest import TestCase
import wave

import numpy as np

from app.core.config import Settings
from app.services.spectral_anti_spoofing_service import (
    SPECTRAL_FEATURE_COUNT,
    SpectralAntiSpoofingService,
    extract_robust_spectral_features,
    extract_spectral_features,
)


class SpectralAntiSpoofingServiceTest(TestCase):
    def test_feature_vector_is_finite_and_stable_size(self):
        sample_rate = 16_000
        time = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
        samples = 0.2 * np.sin(2.0 * math.pi * 220.0 * time)

        features = extract_spectral_features(samples, sample_rate)

        self.assertEqual(features.shape, (SPECTRAL_FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())

    def test_packaged_model_returns_bounded_score(self):
        service = SpectralAntiSpoofingService(Settings())
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "tone.wav"
            sample_rate = 16_000
            pcm = bytearray()
            for index in range(sample_rate * 3):
                value = int(4_000 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate))
                pcm.extend(value.to_bytes(2, byteorder="little", signed=True))
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(bytes(pcm))

            result = service.detect_file(wav_path)

        self.assertGreaterEqual(result.spoof_score, 0.0)
        self.assertLessEqual(result.spoof_score, 1.0)
        self.assertEqual(result.model_name, "isfam/spectral-ensemble-telephone-v2")
        self.assertEqual(result.analyzed_segments, 1)

    def test_robust_features_are_invariant_to_volume_scaling(self):
        sample_rate = 16_000
        time = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
        samples = 0.2 * np.sin(2.0 * math.pi * 220.0 * time)

        normal = extract_robust_spectral_features(samples, sample_rate)
        quiet = extract_robust_spectral_features(samples * 0.1, sample_rate)

        np.testing.assert_allclose(normal, quiet, atol=6e-3, rtol=1e-3)

    def test_runtime_threshold_can_be_overridden(self):
        service = SpectralAntiSpoofingService(Settings(anti_spoofing_threshold=0.75))

        self.assertEqual(service.threshold, 0.75)

    def test_v1_only_ensemble_alert_requires_confirmation(self):
        combined = SpectralAntiSpoofingService._combine_ensemble_scores(
            v1_score=0.9,
            v2_score=0.1,
            v1_threshold=0.5,
            v2_threshold=0.5,
        )

        self.assertEqual(combined, 0.4)

    def test_v2_ensemble_alert_is_automatic_spoof(self):
        combined = SpectralAntiSpoofingService._combine_ensemble_scores(
            v1_score=0.1,
            v2_score=0.9,
            v1_threshold=0.5,
            v2_threshold=0.5,
        )

        self.assertGreaterEqual(combined, 0.5)
