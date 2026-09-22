"""Evaluate unseen-generator and synthetic call-channel robustness.

This script deliberately uses only the existing DFADD calibration/test split.
It does not claim that synthetic degradations replace a real telephone test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_GATE_MIN_SNR_DB = 25.0
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.services.spectral_anti_spoofing_service import (
    ENSEMBLE_FEATURE_VERSION,
    ROBUST_FEATURE_VERSION,
    SpectralAntiSpoofingService,
    extract_robust_spectral_features,
    extract_spectral_features,
)
from scripts.train_spectral_anti_spoofing import read_wav, select_threshold


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: int
    generator: str | None


def collect_examples(root: Path) -> list[AudioExample]:
    examples = [
        AudioExample(path=path, label=0, generator=None)
        for path in sorted((root / "real").glob("*.wav"))
    ]
    for path in sorted((root / "fake").glob("*.wav")):
        generator = path.name.split("__", 1)[0].lower()
        examples.append(AudioExample(path=path, label=1, generator=generator))
    if not examples or {example.label for example in examples} != {0, 1}:
        raise SystemExit(f"Both real and fake wav files are required under {root}")
    return examples


def load_clean_features(
    examples: list[AudioExample],
    feature_version: str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for index, example in enumerate(examples, start=1):
        samples, sample_rate = read_wav(example.path)
        extractor = (
            extract_robust_spectral_features
            if feature_version == ROBUST_FEATURE_VERSION
            else extract_spectral_features
        )
        features.append(extractor(samples, sample_rate))
        labels.append(example.label)
        if index % 200 == 0 or index == len(examples):
            print(f"clean features: {index}/{len(examples)}")
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def train_fold_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    hidden_size: int = 32,
    weight_decay: float = 1e-3,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = (features.std(axis=0) + 1e-5).astype(np.float32)
    normalized = torch.from_numpy((features - feature_mean) / feature_std)
    targets = torch.from_numpy(labels[:, None]).float()
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], hidden_size),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(hidden_size, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(normalized)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, feature_mean, feature_std


def predict_fold(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    normalized = torch.from_numpy((features - feature_mean) / feature_std)
    with torch.inference_mode():
        return torch.sigmoid(model(normalized)).squeeze(1).numpy()


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if labels.size == 0:
        return {
            "samples": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "fake_recall": 0.0,
            "false_positive_rate": 0.0,
        }
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    fake_recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "samples": int(labels.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / labels.size, 6),
        "balanced_accuracy": round((fake_recall + 1.0 - false_positive_rate) / 2.0, 6),
        "fake_recall": round(fake_recall, 6),
        "false_positive_rate": round(false_positive_rate, 6),
    }


def evaluate_generator_holdout(
    calibration_examples: list[AudioExample],
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    test_examples: list[AudioExample],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    max_fpr: float,
    hidden_size: int,
    weight_decay: float,
) -> list[dict[str, object]]:
    generators = sorted(
        {example.generator for example in calibration_examples if example.generator is not None}
    )
    results: list[dict[str, object]] = []
    for fold_index, held_out in enumerate(generators):
        training_mask = np.asarray(
            [
                example.label == 0 or example.generator != held_out
                for example in calibration_examples
            ],
            dtype=bool,
        )
        evaluation_mask = np.asarray(
            [
                example.label == 0 or example.generator == held_out
                for example in test_examples
            ],
            dtype=bool,
        )
        train_x = calibration_features[training_mask]
        train_y = calibration_labels[training_mask]
        test_x = test_features[evaluation_mask]
        test_y = test_labels[evaluation_mask]
        model, feature_mean, feature_std = train_fold_model(
            train_x,
            train_y,
            seed=seed + fold_index,
            epochs=epochs,
            hidden_size=hidden_size,
            weight_decay=weight_decay,
        )
        training_scores = predict_fold(model, train_x, feature_mean, feature_std)
        threshold = select_threshold(training_scores, train_y, max_fpr)
        test_scores = predict_fold(model, test_x, feature_mean, feature_std)
        metrics = classification_metrics(test_scores, test_y, threshold)
        result: dict[str, object] = {
            "held_out_generator": held_out,
            "training_real": int(np.sum(train_y == 0)),
            "training_fake": int(np.sum(train_y == 1)),
            "test_real": int(np.sum(test_y == 0)),
            "test_fake": int(np.sum(test_y == 1)),
            "threshold": round(float(threshold), 8),
            **metrics,
        }
        results.append(result)
        print(
            f"holdout {held_out}: recall={metrics['fake_recall']:.4f}, "
            f"fpr={metrics['false_positive_rate']:.4f}, "
            f"balanced={metrics['balanced_accuracy']:.4f}"
        )
    return results


def evaluate_generator_holdout_ensemble(
    calibration_examples: list[AudioExample],
    calibration_v1: np.ndarray,
    calibration_v2: np.ndarray,
    calibration_labels: np.ndarray,
    test_examples: list[AudioExample],
    test_v1: np.ndarray,
    test_v2: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    max_fpr: float,
) -> list[dict[str, object]]:
    generators = sorted(
        {example.generator for example in calibration_examples if example.generator is not None}
    )
    results: list[dict[str, object]] = []
    for fold_index, held_out in enumerate(generators):
        training_mask = np.asarray(
            [
                example.label == 0 or example.generator != held_out
                for example in calibration_examples
            ],
            dtype=bool,
        )
        evaluation_mask = np.asarray(
            [
                example.label == 0 or example.generator == held_out
                for example in test_examples
            ],
            dtype=bool,
        )
        train_y = calibration_labels[training_mask]
        test_y = test_labels[evaluation_mask]
        component_predictions: list[np.ndarray] = []
        thresholds: dict[str, float] = {}
        for component_name, train_x, test_x, hidden_size, component_seed in (
            (
                "v1",
                calibration_v1[training_mask],
                test_v1[evaluation_mask],
                32,
                seed + fold_index,
            ),
            (
                "v2",
                calibration_v2[training_mask],
                test_v2[evaluation_mask],
                16,
                seed + 100 + fold_index,
            ),
        ):
            model, feature_mean, feature_std = train_fold_model(
                train_x,
                train_y,
                seed=component_seed,
                epochs=epochs,
                hidden_size=hidden_size,
            )
            training_scores = predict_fold(model, train_x, feature_mean, feature_std)
            threshold = select_threshold(training_scores, train_y, max_fpr)
            test_scores = predict_fold(model, test_x, feature_mean, feature_std)
            thresholds[component_name] = round(float(threshold), 8)
            component_predictions.append(test_scores >= threshold)
        predicted = component_predictions[0] | component_predictions[1]
        metrics = classification_metrics(predicted.astype(np.float32), test_y, 0.5)
        result: dict[str, object] = {
            "held_out_generator": held_out,
            "training_real": int(np.sum(train_y == 0)),
            "training_fake": int(np.sum(train_y == 1)),
            "test_real": int(np.sum(test_y == 0)),
            "test_fake": int(np.sum(test_y == 1)),
            "component_thresholds": thresholds,
            **metrics,
        }
        results.append(result)
        print(
            f"ensemble holdout {held_out}: recall={metrics['fake_recall']:.4f}, "
            f"fpr={metrics['false_positive_rate']:.4f}, "
            f"balanced={metrics['balanced_accuracy']:.4f}"
        )
    return results


def _fit_length(samples: np.ndarray, size: int) -> np.ndarray:
    if samples.size >= size:
        return samples[:size].astype(np.float32)
    return np.pad(samples, (0, size - samples.size)).astype(np.float32)


def telephone_8khz(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    downsampled = resample_poly(samples, 1, 2)
    restored = resample_poly(downsampled, 2, 1)
    filtered = sosfilt(
        butter(6, (300, 3400), btype="bandpass", fs=sample_rate, output="sos"),
        restored,
    )
    return _fit_length(filtered, samples.size)


def add_noise(
    samples: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    *,
    colored: bool = False,
) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(samples * samples)))
    if signal_rms <= 1e-8:
        return samples.copy()
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, 1.0, size=samples.size)
    if colored:
        noise = np.convolve(noise, np.asarray([1, 2, 3, 2, 1]) / 9.0, mode="same")
    noise = noise * (noise_rms / max(float(np.sqrt(np.mean(noise * noise))), 1e-8))
    return np.clip(samples + noise, -1.0, 1.0).astype(np.float32)


def add_reverb(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    output = samples.astype(np.float32).copy()
    for delay_seconds, gain in ((0.04, 0.45), (0.09, 0.25), (0.15, 0.15)):
        delay = int(sample_rate * delay_seconds)
        output[delay:] += gain * samples[:-delay]
    peak = float(np.max(np.abs(output)))
    return output / max(1.0, peak)


def packet_loss(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    output = samples.copy()
    frame = max(1, int(sample_rate * 0.02))
    for frame_index, start in enumerate(range(0, output.size, frame)):
        if frame_index % 10 == 4:
            output[start : start + frame] = 0.0
    return output


def strong_clipping(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-8:
        return samples.copy()
    normalized = samples / peak
    return np.clip(normalized * 1.5, -0.60, 0.60).astype(np.float32)


def transform_audio(
    samples: np.ndarray,
    sample_rate: int,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition == "clean":
        return samples
    if condition == "telephone_8khz":
        return telephone_8khz(samples, sample_rate)
    noise_match = re.fullmatch(r"noise_(\d+)db(?:_([a-z0-9]+))?", condition)
    if noise_match:
        return add_noise(
            samples,
            float(noise_match.group(1)),
            rng,
            colored=noise_match.group(2) == "colored",
        )
    volume_match = re.fullmatch(r"low_volume_-(\d+)db", condition)
    if volume_match:
        gain_db = -float(volume_match.group(1))
        return (samples * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    if condition == "strong_clipping":
        return strong_clipping(samples)
    packet_match = re.fullmatch(r"packet_loss_(\d+)pct", condition)
    if packet_match:
        loss_percent = max(1, int(packet_match.group(1)))
        period = max(1, round(100 / loss_percent))
        output = samples.copy()
        frame = max(1, int(sample_rate * 0.02))
        phase = min(4, period - 1)
        for frame_index, start in enumerate(range(0, output.size, frame)):
            if frame_index % period == phase:
                output[start : start + frame] = 0.0
        return output
    if condition == "reverb":
        return add_reverb(samples, sample_rate)
    replay_match = re.fullmatch(r"replay_simulation(?:_(\d+)db)?", condition)
    if replay_match:
        replayed = telephone_8khz(add_reverb(samples, sample_rate), sample_rate)
        snr_db = float(replay_match.group(1) or 15.0)
        return add_noise(
            replayed,
            snr_db,
            rng,
            colored=replay_match.group(1) is not None,
        )
    raise ValueError(f"Unknown condition: {condition}")


def stable_rng(path: Path, condition: str) -> np.random.Generator:
    digest = sha256(f"{path.name}:{condition}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def estimate_dynamic_snr_db(samples: np.ndarray, sample_rate: int) -> float:
    frame_size = max(1, int(sample_rate * 0.02))
    usable = samples[: samples.size - (samples.size % frame_size)]
    if usable.size < frame_size:
        return 0.0
    frames = usable.reshape(-1, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    noise_level = float(np.percentile(frame_rms, 10.0))
    speech_level = float(np.percentile(frame_rms, 90.0))
    return float(20.0 * np.log10((speech_level + 1e-8) / (noise_level + 1e-8)))


def evaluate_call_conditions(
    examples: list[AudioExample],
    *,
    threshold: float,
    model_path: Path,
    model_name: str,
) -> list[dict[str, object]]:
    conditions = (
        "clean",
        "telephone_8khz",
        "noise_20db",
        "noise_10db",
        "low_volume_-18db",
        "strong_clipping",
        "packet_loss_10pct",
        "reverb",
        "replay_simulation",
    )
    settings = Settings(
        anti_spoofing_threshold=threshold,
        spectral_anti_spoofing_model_path=model_path,
        spectral_anti_spoofing_model_name=model_name,
    )
    service = SpectralAntiSpoofingService(settings)
    loaded_audio = [(example, *read_wav(example.path)) for example in examples]
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    results: list[dict[str, object]] = []
    for condition in conditions:
        scores: list[float] = []
        snr_estimates: list[float] = []
        for index, (example, samples, sample_rate) in enumerate(loaded_audio, start=1):
            transformed = transform_audio(
                samples,
                sample_rate,
                condition,
                stable_rng(example.path, condition),
            )
            scores.append(service.predict_samples(transformed, sample_rate))
            snr_estimates.append(estimate_dynamic_snr_db(transformed, sample_rate))
            if index % 200 == 0 or index == len(loaded_audio):
                print(f"condition {condition}: {index}/{len(loaded_audio)}")
        metrics = classification_metrics(np.asarray(scores), labels, threshold)
        scores_array = np.asarray(scores)
        snr_array = np.asarray(snr_estimates)
        quality_mask = snr_array >= QUALITY_GATE_MIN_SNR_DB
        ambiguous_mask = (scores_array >= threshold * 0.7) & (
            scores_array < threshold
        )
        confirmation_mask = ~quality_mask | ambiguous_mask
        catch_or_confirm_mask = scores_array >= threshold * 0.7
        gated_metrics = classification_metrics(
            scores_array[quality_mask],
            labels[quality_mask],
            threshold,
        )
        real_mask = labels == 0
        fake_mask = labels == 1
        results.append(
            {
                "condition": condition,
                **metrics,
                "estimated_snr_median_db": round(float(np.median(snr_estimates)), 3),
                "estimated_snr_p10_db": round(float(np.percentile(snr_estimates, 10)), 3),
                "estimated_snr_p90_db": round(float(np.percentile(snr_estimates, 90)), 3),
                "quality_gate_min_snr_db": QUALITY_GATE_MIN_SNR_DB,
                "auto_analyzed_rate": round(float(np.mean(quality_mask)), 6),
                "real_auto_analyzed_rate": round(
                    float(np.mean(quality_mask[real_mask])), 6
                ),
                "fake_auto_analyzed_rate": round(
                    float(np.mean(quality_mask[fake_mask])), 6
                ),
                "additional_confirmation_rate": round(
                    float(np.mean(confirmation_mask)), 6
                ),
                "fake_catch_or_confirm_rate": round(
                    float(
                        np.mean(
                            (~quality_mask[fake_mask])
                            | catch_or_confirm_mask[fake_mask]
                        )
                    ),
                    6,
                ),
                "real_auto_safe_rate": round(
                    float(
                        np.mean(
                            quality_mask[real_mask]
                            & ~catch_or_confirm_mask[real_mask]
                        )
                    ),
                    6,
                ),
                "gated_metrics": gated_metrics,
            }
        )
        print(
            f"condition {condition}: recall={metrics['fake_recall']:.4f}, "
            f"fpr={metrics['false_positive_rate']:.4f}, "
            f"balanced={metrics['balanced_accuracy']:.4f}"
        )
    return results


def summarize_holdout(results: list[dict[str, object]]) -> dict[str, float]:
    recalls = [float(result["fake_recall"]) for result in results]
    fprs = [float(result["false_positive_rate"]) for result in results]
    balanced = [float(result["balanced_accuracy"]) for result in results]
    return {
        "mean_fake_recall": round(float(np.mean(recalls)), 6),
        "min_fake_recall": round(float(np.min(recalls)), 6),
        "max_fake_recall": round(float(np.max(recalls)), 6),
        "mean_false_positive_rate": round(float(np.mean(fprs)), 6),
        "max_false_positive_rate": round(float(np.max(fprs)), 6),
        "mean_balanced_accuracy": round(float(np.mean(balanced)), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "public" / "dfadd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "spectral_robustness_results.json",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_v1.npz",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--feature-version",
        choices=("spectral_v1", ROBUST_FEATURE_VERSION, ENSEMBLE_FEATURE_VERSION),
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-conditions", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    model_path = args.model_path.resolve()
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(metadata["threshold"])
    )
    model_name = args.model_name or str(metadata["model_name"])
    feature_version = args.feature_version or str(
        metadata.get("feature_version", "spectral_v1")
    )
    calibration_examples = collect_examples(args.dataset.resolve() / "calibration")
    test_examples = collect_examples(args.dataset.resolve() / "test")
    if feature_version == ENSEMBLE_FEATURE_VERSION:
        calibration_features, calibration_labels = load_clean_features(
            calibration_examples,
            "spectral_v1",
        )
        calibration_robust_features, _ = load_clean_features(
            calibration_examples,
            ROBUST_FEATURE_VERSION,
        )
        test_features, test_labels = load_clean_features(test_examples, "spectral_v1")
        test_robust_features, _ = load_clean_features(
            test_examples,
            ROBUST_FEATURE_VERSION,
        )
    else:
        calibration_features, calibration_labels = load_clean_features(
            calibration_examples,
            feature_version,
        )
        test_features, test_labels = load_clean_features(test_examples, feature_version)
    holdout_results = []
    if not args.skip_holdout:
        if feature_version == ENSEMBLE_FEATURE_VERSION:
            holdout_results = evaluate_generator_holdout_ensemble(
                calibration_examples,
                calibration_features,
                calibration_robust_features,
                calibration_labels,
                test_examples,
                test_features,
                test_robust_features,
                test_labels,
                seed=args.seed,
                epochs=args.epochs,
                max_fpr=args.max_fpr,
            )
        else:
            holdout_results = evaluate_generator_holdout(
                calibration_examples,
                calibration_features,
                calibration_labels,
                test_examples,
                test_features,
                test_labels,
                seed=args.seed,
                epochs=args.epochs,
                max_fpr=args.max_fpr,
                hidden_size=args.hidden_size,
                weight_decay=args.weight_decay,
            )
    condition_results = []
    if not args.skip_conditions:
        condition_results = evaluate_call_conditions(
            test_examples,
            threshold=threshold,
            model_path=model_path,
            model_name=model_name,
        )
    payload = {
        "dataset": str(args.dataset.resolve()),
        "policy": {
            "generator_holdout": "train on four fake generators; test on the fifth",
            "real_split": "200 calibration real for training; 400 test real for evaluation",
            "threshold_selection": "training-fold threshold under configured real FPR cap",
            "synthetic_conditions": "deterministic simulations, not real telephone recordings",
            "low_snr_policy": (
                f"estimated SNR below {QUALITY_GATE_MIN_SNR_DB:g} dB requires "
                "another voice sample"
            ),
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "max_fpr": args.max_fpr,
        "model_name": model_name,
        "model_path": str(model_path),
        "feature_version": feature_version,
        "deployed_threshold": threshold,
        "generator_holdout": holdout_results,
        "generator_holdout_summary": (
            summarize_holdout(holdout_results) if holdout_results else None
        ),
        "call_conditions": condition_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved robustness results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
