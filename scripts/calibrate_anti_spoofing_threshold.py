"""Select and report an anti-spoof threshold under a real-speech FPR cap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise SystemExit(f"No score rows found in {path}")
    labels = {row["label"] for row in rows}
    if labels != {"real", "fake"}:
        raise SystemExit(f"Both real and fake rows are required, found: {sorted(labels)}")
    return rows


def metrics(rows: list[dict[str, str]], threshold: float) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in rows:
        predicted_fake = float(row["spoof_score"]) >= threshold
        actual_fake = row["label"] == "fake"
        if predicted_fake and actual_fake:
            tp += 1
        elif predicted_fake:
            fp += 1
        elif actual_fake:
            fn += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    accuracy = (tp + tn) / len(rows)
    balanced_accuracy = (recall + (1.0 - fpr)) / 2.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 8),
        "samples": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "precision": round(precision, 6),
        "fake_recall": round(recall, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
    }


def candidate_metrics(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    scores = sorted({float(row["spoof_score"]) for row in rows})
    thresholds = [0.0, *scores, min(1.000001, scores[-1] + 0.000001)]
    return [metrics(rows, threshold) for threshold in thresholds]


def choose_threshold(
    candidates: list[dict[str, float | int]],
    max_fpr: float,
) -> dict[str, float | int]:
    eligible = [row for row in candidates if float(row["false_positive_rate"]) <= max_fpr]
    if not eligible:
        raise SystemExit(f"No threshold satisfies max FPR {max_fpr}")
    return max(
        eligible,
        key=lambda row: (
            float(row["fake_recall"]),
            float(row["balanced_accuracy"]),
            -float(row["false_positive_rate"]),
            float(row["threshold"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.max_fpr <= 1.0:
        raise SystemExit("--max-fpr must be between 0 and 1")

    rows = load_rows(args.scores.resolve())
    candidates = candidate_metrics(rows)
    selected = choose_threshold(candidates, args.max_fpr)
    eer = min(
        candidates,
        key=lambda row: abs(
            float(row["false_positive_rate"]) - float(row["false_negative_rate"])
        ),
    )
    payload = {
        "source_scores": str(args.scores.resolve()),
        "selection_policy": "maximize fake recall under the real-speech FPR cap",
        "max_fpr": args.max_fpr,
        "selected": selected,
        "approximate_eer": round(
            (float(eer["false_positive_rate"]) + float(eer["false_negative_rate"])) / 2.0,
            6,
        ),
        "eer_threshold": eer["threshold"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
