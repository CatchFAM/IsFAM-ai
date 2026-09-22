from unittest import TestCase

import numpy as np

from scripts.evaluate_spectral_robustness import (
    add_noise,
    classification_metrics,
    packet_loss,
    telephone_8khz,
)


class SpectralRobustnessTest(TestCase):
    def test_transforms_preserve_length_and_finite_values(self):
        samples = np.sin(np.linspace(0, 100, 16_000, dtype=np.float32)) * 0.2
        rng = np.random.default_rng(7)

        for transformed in (
            telephone_8khz(samples, 16_000),
            add_noise(samples, 10.0, rng),
            packet_loss(samples, 16_000),
        ):
            self.assertEqual(transformed.shape, samples.shape)
            self.assertTrue(np.isfinite(transformed).all())

    def test_classification_metrics_count_errors(self):
        metrics = classification_metrics(
            np.asarray([0.1, 0.8, 0.9, 0.2]),
            np.asarray([0, 0, 1, 1]),
            threshold=0.5,
        )

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
