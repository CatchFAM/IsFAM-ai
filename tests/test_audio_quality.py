import math
from unittest import TestCase

import numpy as np

from app.utils.audio_quality import _compute_estimated_snr_db


class AudioQualityTest(TestCase):
    def test_dynamic_snr_separates_clean_speech_from_heavy_noise(self):
        sample_rate = 16_000
        time = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
        envelope = np.where((time % 0.6) < 0.4, 1.0, 0.02)
        clean = 0.2 * np.sin(2.0 * math.pi * 220.0 * time) * envelope
        noisy = clean + np.random.default_rng(4).normal(0.0, 0.08, clean.size)

        clean_snr = _compute_estimated_snr_db(clean.tolist(), sample_rate)
        noisy_snr = _compute_estimated_snr_db(noisy.tolist(), sample_rate)

        self.assertGreater(clean_snr, 22.0)
        self.assertLess(noisy_snr, 22.0)
