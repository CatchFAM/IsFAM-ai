"""Train a call-channel-augmented spectral anti-spoofing candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.spectral_anti_spoofing_service import (
    ROBUST_FEATURE_VERSION,
    extract_robust_spectral_features,
)
from scripts.evaluate_spectral_robustness import (
    collect_examples,
    stable_rng,
    train_fold_model,
    transform_audio,
)
from scripts.train_spectral_anti_spoofing import (
    compute_metrics,
    read_wav,
    select_threshold,
)


TRAINING_CONDITIONS = (
    "clean",
    "telephone_8khz",
    "noise_30db_a",
    "noise_25db_a",
    "noise_20db",
    "noise_20db_b",
    "noise_15db_a",
    "noise_15db_b",
    "noise_10db",
    "noise_10db_b",
    "noise_5db_a",
    "noise_20db_colored",
    "noise_10db_colored",
    "low_volume_-6db",
    "low_volume_-12db",
    "low_volume_-18db",
    "low_volume_-24db",
    "strong_clipping",
    "packet_loss_5pct",
    "packet_loss_10pct",
    "packet_loss_20pct",
    "reverb",
    "replay_simulation",
    "replay_simulation_20db",
    "replay_simulation_15db",
    "replay_simulation_10db",
)


def content_group(path: Path) -> str:
    match = re.search(r"p\d+_\d+", path.name.lower())
    return match.group(0) if match else path.stem.lower()


def split_examples(
    examples: list,
    validation_percent: int,
) -> tuple[list, list]:
    training = []
    validation = []
    for example in examples:
        digest = sha256(
            f"robust-v2-validation:{content_group(example.path)}".encode("utf-8")
        ).hexdigest()
        target = validation if int(digest[:8], 16) % 100 < validation_percent else training
        target.append(example)
    if {example.label for example in training} != {0, 1}:
        raise SystemExit("Training split must contain both labels")
    if {example.label for example in validation} != {0, 1}:
        raise SystemExit("Validation split must contain both labels")
    return training, validation


def build_augmented_features(
    examples: list,
) -> tuple[np.ndarray, np.ndarray, list[Path], list[str]]:
    loaded = [(example, *read_wav(example.path)) for example in examples]
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    conditions: list[str] = []
    total = len(loaded) * len(TRAINING_CONDITIONS)
    completed = 0
    for condition in TRAINING_CONDITIONS:
        for example, samples, sample_rate in loaded:
            transformed = transform_audio(
                samples,
                sample_rate,
                condition,
                stable_rng(example.path, condition),
            )
            features.append(extract_robust_spectral_features(transformed, sample_rate))
            labels.append(example.label)
            paths.append(example.path)
            conditions.append(condition)
            completed += 1
            if completed % 400 == 0 or completed == total:
                print(f"augmented features: {completed}/{total}")
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        paths,
        conditions,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "public" / "dfadd" / "calibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_v2.npz",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--validation-percent", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    torch.set_num_threads(1)
    all_examples = collect_examples(args.dataset.resolve())
    training_examples, validation_examples = split_examples(
        all_examples,
        args.validation_percent,
    )
    print(
        f"split: train={len(training_examples)}, validation={len(validation_examples)}"
    )
    features, labels, paths, conditions = build_augmented_features(training_examples)
    validation_features, validation_labels, validation_paths, validation_conditions = (
        build_augmented_features(validation_examples)
    )
    model, feature_mean, feature_std = train_fold_model(
        features,
        labels,
        seed=args.seed,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        weight_decay=args.weight_decay,
    )
    normalized = torch.from_numpy((features - feature_mean) / feature_std)
    normalized_validation = torch.from_numpy(
        (validation_features - feature_mean) / feature_std
    )
    with torch.inference_mode():
        scores = torch.sigmoid(model(normalized)).squeeze(1).numpy()
        validation_scores = torch.sigmoid(model(normalized_validation)).squeeze(1).numpy()
    threshold = select_threshold(validation_scores, validation_labels, args.max_fpr)
    training_metrics = compute_metrics(scores, labels, threshold)
    validation_metrics = compute_metrics(validation_scores, validation_labels, threshold)
    hidden = model[0]
    output = model[2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mean=feature_mean,
        feature_std=feature_std,
        hidden_weight=hidden.weight.detach().numpy().astype(np.float32),
        hidden_bias=hidden.bias.detach().numpy().astype(np.float32),
        output_weight=output.weight.detach().numpy().astype(np.float32),
        output_bias=output.bias.detach().numpy().astype(np.float32),
    )
    dataset_digest = sha256()
    for path, condition in zip(
        paths + validation_paths,
        conditions + validation_conditions,
    ):
        dataset_digest.update(path.name.encode("utf-8"))
        dataset_digest.update(condition.encode("utf-8"))
        dataset_digest.update(path.read_bytes())
    metadata = {
        "model_name": "isfam/spectral-mlp-telephone-robust-v2",
        "feature_version": ROBUST_FEATURE_VERSION,
        "training_dataset": "DFADD deterministic calibration subset with call augmentation",
        "dataset_sha256": dataset_digest.hexdigest(),
        "seed": args.seed,
        "epochs": args.epochs,
        "max_real_fpr": args.max_fpr,
        "validation_percent": args.validation_percent,
        "hidden_size": args.hidden_size,
        "weight_decay": args.weight_decay,
        "threshold": threshold,
        "feature_count": int(features.shape[1]),
        "training_examples": int(features.shape[0]),
        "validation_examples": int(validation_features.shape[0]),
        "augmentation_conditions": list(TRAINING_CONDITIONS),
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
