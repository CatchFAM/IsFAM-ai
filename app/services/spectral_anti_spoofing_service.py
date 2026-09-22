import array
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.anti_spoofing_service import (
    AntiSpoofingError,
    AntiSpoofingResult,
    LabelScore,
)


SPECTRAL_FEATURE_COUNT = 196
ROBUST_FEATURE_VERSION = "telephone_robust_v2"
ENSEMBLE_FEATURE_VERSION = "spectral_ensemble_v2"


def extract_spectral_features(
    samples: np.ndarray,
    sample_rate: int,
    max_seconds: float = 5.0,
) -> np.ndarray:
    """Extract inexpensive time/frequency statistics from one call chunk."""

    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    max_samples = max(1, int(sample_rate * max_seconds))
    if waveform.size > max_samples:
        start = (waveform.size - max_samples) // 2
        waveform = waveform[start : start + max_samples]

    frame_size = max(1, int(sample_rate * 0.025))
    hop_size = max(1, int(sample_rate * 0.010))
    if waveform.size < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.size))

    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_size)[::hop_size]
    windowed = frames * np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(windowed, n=512)) + 1e-7
    log_spectrum = np.log(spectrum)
    frequency_groups = np.array_split(np.arange(log_spectrum.shape[1]), 64)
    pooled = np.stack(
        [log_spectrum[:, group].mean(axis=1) for group in frequency_groups],
        axis=1,
    )
    deltas = np.diff(pooled, axis=0) if len(pooled) > 1 else np.zeros_like(pooled)

    zero_crossing_rate = (
        float(np.mean(waveform[:-1] * waveform[1:] < 0)) if waveform.size > 1 else 0.0
    )
    rms_energy = float(np.sqrt(np.mean(waveform * waveform)))
    peak_amplitude = float(np.max(np.abs(waveform)))
    duration_seconds = waveform.size / float(sample_rate)
    features = np.concatenate(
        [
            pooled.mean(axis=0),
            pooled.std(axis=0),
            deltas.std(axis=0),
            np.asarray(
                [zero_crossing_rate, rms_energy, peak_amplitude, duration_seconds],
                dtype=np.float64,
            ),
        ]
    ).astype(np.float32)
    if features.shape != (SPECTRAL_FEATURE_COUNT,):
        raise AntiSpoofingError(f"unexpected spectral feature shape: {features.shape}")
    return features


def extract_robust_spectral_features(
    samples: np.ndarray,
    sample_rate: int,
    max_seconds: float = 5.0,
) -> np.ndarray:
    """Extract scale-invariant features from the stable telephone speech band."""

    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    max_samples = max(1, int(sample_rate * max_seconds))
    if waveform.size > max_samples:
        start = (waveform.size - max_samples) // 2
        waveform = waveform[start : start + max_samples]
    waveform = waveform - float(np.mean(waveform))
    rms = float(np.sqrt(np.mean(waveform * waveform)))
    if rms > 1e-7:
        waveform = waveform * (0.1 / rms)
    waveform = np.clip(waveform, -1.0, 1.0)

    frame_size = max(1, int(sample_rate * 0.025))
    hop_size = max(1, int(sample_rate * 0.010))
    if waveform.size < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.size))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame_size)[::hop_size]
    windowed = frames * np.hanning(frame_size)
    spectrum = np.abs(np.fft.rfft(windowed, n=512)) + 1e-7
    frequencies = np.fft.rfftfreq(512, d=1.0 / sample_rate)
    speech_band = (frequencies >= 400.0) & (frequencies <= 3200.0)
    log_spectrum = np.log(spectrum[:, speech_band])
    # Remove frame loudness. The classifier should focus on spectral shape,
    # not microphone gain or call volume.
    log_spectrum = log_spectrum - log_spectrum.mean(axis=1, keepdims=True)
    frequency_groups = np.array_split(np.arange(log_spectrum.shape[1]), 64)
    pooled = np.stack(
        [log_spectrum[:, group].mean(axis=1) for group in frequency_groups],
        axis=1,
    )
    deltas = np.diff(pooled, axis=0) if len(pooled) > 1 else np.zeros_like(pooled)

    zero_crossing_rate = (
        float(np.mean(waveform[:-1] * waveform[1:] < 0)) if waveform.size > 1 else 0.0
    )
    normalized_rms = float(np.sqrt(np.mean(waveform * waveform)))
    crest_factor = float(np.max(np.abs(waveform))) / max(normalized_rms, 1e-7)
    band_power = spectrum[:, speech_band] ** 2
    spectral_flatness = float(
        np.mean(
            np.exp(np.mean(np.log(band_power + 1e-12), axis=1))
            / (np.mean(band_power, axis=1) + 1e-12)
        )
    )
    duration_seconds = waveform.size / float(sample_rate)
    features = np.concatenate(
        [
            pooled.mean(axis=0),
            pooled.std(axis=0),
            deltas.std(axis=0),
            np.asarray(
                [zero_crossing_rate, crest_factor, spectral_flatness, duration_seconds],
                dtype=np.float64,
            ),
        ]
    ).astype(np.float32)
    if features.shape != (SPECTRAL_FEATURE_COUNT,):
        raise AntiSpoofingError(f"unexpected robust feature shape: {features.shape}")
    return features


class SpectralAntiSpoofingService:
    """Small calibrated MLP for fast synthetic-speech screening on CPU."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.spectral_anti_spoofing_model_name
        self.target_sample_rate = settings.target_sample_rate
        self.window_seconds = settings.anti_spoofing_window_seconds
        self.max_audio_seconds = self.window_seconds
        self.hop_seconds = settings.anti_spoofing_hop_seconds
        self.batch_size = 1
        self.device = "cpu"
        self._is_warmed_up = False

        artifact_path = settings.spectral_anti_spoofing_model_path
        metadata_path = artifact_path.with_suffix(".json")
        if not artifact_path.exists() or not metadata_path.exists():
            raise AntiSpoofingError(
                f"spectral anti-spoof artifact is missing: {artifact_path}. "
                "Run scripts/train_spectral_anti_spoofing.py"
            )
        try:
            artifact = np.load(artifact_path, allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.feature_version = str(metadata.get("feature_version", "spectral_v1"))
            if self.feature_version == ENSEMBLE_FEATURE_VERSION:
                self._load_ensemble_artifact(artifact, metadata)
            else:
                self.feature_mean = artifact["feature_mean"].astype(np.float32)
                self.feature_std = artifact["feature_std"].astype(np.float32)
                self.hidden_weight = artifact["hidden_weight"].astype(np.float32)
                self.hidden_bias = artifact["hidden_bias"].astype(np.float32)
                self.output_weight = artifact["output_weight"].astype(np.float32)
                self.output_bias = artifact["output_bias"].astype(np.float32)
        except Exception as exc:
            raise AntiSpoofingError("failed to load spectral anti-spoof artifact") from exc

        self.threshold = settings.anti_spoofing_threshold
        if (
            self.feature_version != ENSEMBLE_FEATURE_VERSION
            and self.feature_mean.shape != (SPECTRAL_FEATURE_COUNT,)
        ):
            raise AntiSpoofingError("spectral artifact feature count does not match runtime")
        if int(metadata.get("feature_count", -1)) != SPECTRAL_FEATURE_COUNT:
            raise AntiSpoofingError("spectral metadata feature count does not match runtime")

    def warm_up(self) -> None:
        if self._is_warmed_up:
            return
        empty_features = np.zeros(SPECTRAL_FEATURE_COUNT, dtype=np.float32)
        if self.feature_version == ENSEMBLE_FEATURE_VERSION:
            self._predict_component_score("v1", empty_features)
            self._predict_component_score("v2", empty_features)
        else:
            self._predict_score(empty_features)
        self._is_warmed_up = True

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def detect_file(self, wav_path: Path) -> AntiSpoofingResult:
        samples = self._load_standard_wav_samples(wav_path)
        raw_score = self.predict_samples(samples, self.target_sample_rate)
        spoof_score = round(raw_score, 4)
        is_spoofed = raw_score >= self.threshold
        is_ambiguous = not is_spoofed and raw_score >= self.threshold * 0.7
        predicted_label = (
            "fake" if is_spoofed else ("uncertain" if is_ambiguous else "real")
        )
        predicted_score = raw_score if is_spoofed or is_ambiguous else 1.0 - raw_score
        return AntiSpoofingResult(
            is_spoofed=is_spoofed,
            spoof_score=spoof_score,
            threshold=round(self.threshold, 4),
            predicted_label=predicted_label,
            predicted_score=round(predicted_score, 4),
            message=(
                "spoof"
                if is_spoofed
                else ("additional_confirmation" if is_ambiguous else "bonafide")
            ),
            model_name=self.model_name,
            analyzed_segments=1,
            max_spoof_segment_index=0,
            segment_seconds=self.window_seconds,
            label_scores=[
                LabelScore(label="real", score=round(1.0 - raw_score, 4)),
                LabelScore(label="fake", score=spoof_score),
            ],
        )

    def extract_features(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.feature_version == ROBUST_FEATURE_VERSION:
            return extract_robust_spectral_features(
                samples,
                sample_rate=sample_rate,
                max_seconds=self.window_seconds,
            )
        return extract_spectral_features(
            samples,
            sample_rate=sample_rate,
            max_seconds=self.window_seconds,
        )

    def predict_samples(self, samples: np.ndarray, sample_rate: int) -> float:
        if self.feature_version == ENSEMBLE_FEATURE_VERSION:
            v1_features = extract_spectral_features(
                samples,
                sample_rate=sample_rate,
                max_seconds=self.window_seconds,
            )
            v2_features = extract_robust_spectral_features(
                samples,
                sample_rate=sample_rate,
                max_seconds=self.window_seconds,
            )
            v1_score = self._predict_component_score("v1", v1_features)
            v2_score = self._predict_component_score("v2", v2_features)
            return self._combine_ensemble_scores(
                v1_score,
                v2_score,
                self.component_thresholds["v1"],
                self.component_thresholds["v2"],
            )
        return self._predict_score(self.extract_features(samples, sample_rate))

    def _predict_score(self, features: np.ndarray) -> float:
        return self._predict_with_weights(
            features,
            self.feature_mean,
            self.feature_std,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
        )

    @staticmethod
    def _predict_with_weights(
        features: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        hidden_weight: np.ndarray,
        hidden_bias: np.ndarray,
        output_weight: np.ndarray,
        output_bias: np.ndarray,
    ) -> float:
        normalized = (features - feature_mean) / feature_std
        hidden = normalized @ hidden_weight.T + hidden_bias
        # Match torch.nn.GELU(approximate="tanh") used by the training script.
        hidden = 0.5 * hidden * (
            1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden**3))
        )
        logit = float(hidden @ output_weight.reshape(-1) + output_bias.reshape(-1)[0])
        if logit >= 0:
            return float(1.0 / (1.0 + math.exp(-logit)))
        exp_logit = math.exp(logit)
        return float(exp_logit / (1.0 + exp_logit))

    def _load_ensemble_artifact(self, artifact, metadata: dict) -> None:
        self.component_weights: dict[str, tuple[np.ndarray, ...]] = {}
        self.component_thresholds: dict[str, float] = {}
        for component in metadata.get("components", []):
            prefix = str(component["prefix"])
            weights = (
                artifact[f"{prefix}_feature_mean"].astype(np.float32),
                artifact[f"{prefix}_feature_std"].astype(np.float32),
                artifact[f"{prefix}_hidden_weight"].astype(np.float32),
                artifact[f"{prefix}_hidden_bias"].astype(np.float32),
                artifact[f"{prefix}_output_weight"].astype(np.float32),
                artifact[f"{prefix}_output_bias"].astype(np.float32),
            )
            if weights[0].shape != (SPECTRAL_FEATURE_COUNT,):
                raise AntiSpoofingError("ensemble component feature count does not match runtime")
            self.component_weights[prefix] = weights
            self.component_thresholds[prefix] = float(component["threshold"])
        if set(self.component_weights) != {"v1", "v2"}:
            raise AntiSpoofingError("spectral ensemble requires v1 and v2 components")

    def _predict_component_score(self, prefix: str, features: np.ndarray) -> float:
        return self._predict_with_weights(features, *self.component_weights[prefix])

    @staticmethod
    def _align_component_score(score: float, threshold: float) -> float:
        if score < threshold:
            return 0.5 * score / max(threshold, 1e-8)
        return 0.5 + 0.5 * (score - threshold) / max(1.0 - threshold, 1e-8)

    @classmethod
    def _combine_ensemble_scores(
        cls,
        v1_score: float,
        v2_score: float,
        v1_threshold: float,
        v2_threshold: float,
    ) -> float:
        """Use robust v2 for auto-blocking and v1-only alerts for confirmation."""

        v1_aligned = cls._align_component_score(v1_score, v1_threshold)
        v2_aligned = cls._align_component_score(v2_score, v2_threshold)
        if v2_aligned >= 0.5:
            return max(v1_aligned, v2_aligned)
        if v1_aligned >= 0.5:
            return 0.4
        return max(v1_aligned, v2_aligned)

    def _load_standard_wav_samples(self, wav_path: Path) -> np.ndarray:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                raw_audio = wav_file.readframes(wav_file.getnframes())
        except Exception as exc:
            raise AntiSpoofingError("failed to read wav for spectral anti-spoofing") from exc
        if sample_rate != self.target_sample_rate or sample_width != 2 or not raw_audio:
            raise AntiSpoofingError("spectral anti-spoofing expects non-empty 16 kHz PCM16 wav")

        pcm = array.array("h")
        pcm.frombytes(raw_audio)
        if sys.byteorder == "big":
            pcm.byteswap()
        samples = np.asarray(pcm, dtype=np.float32)
        if channel_count > 1:
            samples = samples.reshape(-1, channel_count).mean(axis=1)
        return samples / 32768.0
