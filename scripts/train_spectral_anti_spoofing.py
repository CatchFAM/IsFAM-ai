"""Train the small CPU anti-spoofing model from the fixed DFADD calibration set."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import wave

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.spectral_anti_spoofing_service import extract_spectral_features


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        raw = wav_file.readframes(wav_file.getnframes())
    if width != 2:
        raise ValueError(f"Expected PCM16 wav: {path}")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples / 32768.0, sample_rate


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    for label_name, label in (("real", 0), ("fake", 1)):
        for path in sorted((root / label_name).glob("*.wav")):
            samples, sample_rate = read_wav(path)
            features.append(extract_spectral_features(samples, sample_rate))
            labels.append(label)
            paths.append(path)
    if not features or set(labels) != {0, 1}:
        raise SystemExit(f"Balanced real/fake calibration files are required under {root}")
    return np.stack(features), np.asarray(labels, dtype=np.int64), paths


def select_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float) -> float:
    max_real = float(np.max(scores[labels == 0]))
    min_fake = float(np.min(scores[labels == 1]))
    if max_real < min_fake:
        # Center the threshold in the observed separation gap instead of placing
        # it on a class boundary. This leaves margin for domain shift.
        return (max_real + min_fake) / 2.0

    choices: list[tuple[float, float, float]] = []
    for threshold in np.unique(scores):
        fpr = float(np.mean(scores[labels == 0] >= threshold))
        recall = float(np.mean(scores[labels == 1] >= threshold))
        if fpr <= max_fpr:
            choices.append((recall, -fpr, float(threshold)))
    if not choices:
        raise SystemExit("No threshold satisfies the requested real-speech FPR cap")
    return max(choices)[2]


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return {
        "samples": int(labels.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / labels.size, 6),
        "fake_recall": round(tp / (tp + fn), 6),
        "false_positive_rate": round(fp / (fp + tn), 6),
    }


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
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_v1.npz",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    features, labels, paths = load_dataset(args.dataset.resolve())
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_std = (features.std(axis=0) + 1e-5).astype(np.float32)
    normalized = torch.from_numpy((features - feature_mean) / feature_std)
    targets = torch.from_numpy(labels[:, None]).float()

    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 32),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(normalized)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(normalized)).squeeze(1).numpy()
    threshold = select_threshold(scores, labels, args.max_fpr)
    metrics = compute_metrics(scores, labels, threshold)
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
    for path in paths:
        dataset_digest.update(path.name.encode("utf-8"))
        dataset_digest.update(path.read_bytes())
    metadata = {
        "model_name": "isfam/spectral-mlp-dfadd-v1",
        "training_dataset": "DFADD deterministic calibration subset",
        "dataset_sha256": dataset_digest.hexdigest(),
        "seed": args.seed,
        "epochs": args.epochs,
        "max_real_fpr": args.max_fpr,
        "threshold": threshold,
        "feature_count": int(features.shape[1]),
        "calibration_metrics": metrics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
