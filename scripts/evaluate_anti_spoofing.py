import argparse
import csv
import json
from statistics import mean, median
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.model_provider import get_anti_spoofing_service
from app.utils.audio import cleanup_temp_files, convert_audio_to_standard_wav


DEFAULT_THRESHOLDS = "0.03,0.05,0.07,0.10,0.20,0.30,0.50"


def parse_thresholds(raw_value: str) -> list[float]:
    thresholds: list[float] = []
    for item in raw_value.split(","):
        value = item.strip()
        if not value:
            continue
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def collect_audio_files(
    dataset_dir: Path,
    allowed_extensions: tuple[str, ...],
    single_label: str | None = None,
) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    allowed = {extension.lower().lstrip(".") for extension in allowed_extensions}

    if single_label is not None:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file() and path.suffix.lower().lstrip(".") in allowed:
                rows.append((path, single_label))
        return rows

    for label in ("real", "fake"):
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower().lstrip(".") in allowed:
                rows.append((path, label))

    return rows


def is_correct(label: str, is_spoofed: bool) -> bool:
    expected_spoofed = label == "fake"
    return expected_spoofed == is_spoofed


def compute_metrics(rows: list[dict[str, str]], threshold: float) -> dict[str, object]:
    tp = tn = fp = fn = 0

    for row in rows:
        label = row["label"]
        prediction_spoofed = float(row["spoof_score"]) >= threshold
        expected_spoofed = label == "fake"

        if prediction_spoofed and expected_spoofed:
            tp += 1
        elif prediction_spoofed and not expected_spoofed:
            fp += 1
        elif not prediction_spoofed and expected_spoofed:
            fn += 1
        else:
            tn += 1

    total = tp + tn + fp + fn
    correct = tp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0

    return {
        "threshold": threshold,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def evaluate(args: argparse.Namespace) -> None:
    settings = get_settings()
    dataset_dir = args.dataset.resolve()
    output_path = args.output.resolve()
    metrics_output_path = args.metrics_output.resolve()
    thresholds = parse_thresholds(args.thresholds)

    files = collect_audio_files(
        dataset_dir,
        settings.allowed_audio_extensions,
        single_label=args.single_label,
    )
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(
            f"No audio files found. Put files under {dataset_dir}/real and {dataset_dir}/fake"
        )

    service = get_anti_spoofing_service()
    if not args.no_warmup:
        service.warm_up()
    temp_paths: list[Path | None] = []
    result_rows: list[dict[str, object]] = []
    evaluation_started_at = perf_counter()

    try:
        for index, (audio_path, label) in enumerate(files, start=1):
            if not args.quiet or index == 1 or index % 25 == 0 or index == len(files):
                print(f"[{index}/{len(files)}] anti-spoofing: {audio_path}")
            end_to_end_started_at = perf_counter()
            wav_path = convert_audio_to_standard_wav(
                input_path=audio_path,
                target_sample_rate=settings.target_sample_rate,
                min_audio_seconds=settings.min_audio_seconds,
            )
            temp_paths.append(wav_path)

            inference_started_at = perf_counter()
            result = service.detect_file(wav_path)
            inference_time_ms = (perf_counter() - inference_started_at) * 1000.0
            end_to_end_time_ms = (perf_counter() - end_to_end_started_at) * 1000.0
            row = {
                "file": str(audio_path.relative_to(dataset_dir)),
                "label": label,
                "spoof_score": result.spoof_score,
                "configured_threshold": result.threshold,
                "is_spoofed": result.is_spoofed,
                "correct": is_correct(label, result.is_spoofed),
                "predicted_label": result.predicted_label,
                "predicted_score": result.predicted_score,
                "model_message": result.message,
                "analyzed_segments": result.analyzed_segments,
                "max_spoof_segment_index": result.max_spoof_segment_index,
                "segment_seconds": result.segment_seconds,
                "model_name": result.model_name,
                "inference_time_ms": round(inference_time_ms, 3),
                "end_to_end_time_ms": round(end_to_end_time_ms, 3),
                "label_scores_json": json.dumps(
                    [
                        {"label": label_score.label, "score": label_score.score}
                        for label_score in result.label_scores
                    ],
                    ensure_ascii=False,
                ),
            }
            result_rows.append(row)
    finally:
        cleanup_temp_files(temp_paths)

    result_fields = [
        "file",
        "label",
        "spoof_score",
        "configured_threshold",
        "is_spoofed",
        "correct",
        "predicted_label",
        "predicted_score",
        "model_message",
        "analyzed_segments",
        "max_spoof_segment_index",
        "segment_seconds",
        "model_name",
        "inference_time_ms",
        "end_to_end_time_ms",
        "label_scores_json",
    ]
    write_csv(output_path, result_rows, result_fields)

    metric_rows = [compute_metrics(result_rows, threshold) for threshold in thresholds]
    metric_fields = [
        "threshold",
        "total",
        "correct",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
    ]
    write_csv(metrics_output_path, metric_rows, metric_fields)

    inference_times = [float(row["inference_time_ms"]) for row in result_rows]
    end_to_end_times = [float(row["end_to_end_time_ms"]) for row in result_rows]
    wall_seconds = perf_counter() - evaluation_started_at
    summary = {
        "dataset": str(dataset_dir),
        "model_name": service.model_name,
        "configured_threshold": service.threshold,
        "samples": len(result_rows),
        "real_samples": sum(row["label"] == "real" for row in result_rows),
        "fake_samples": sum(row["label"] == "fake" for row in result_rows),
        "warmup_enabled": not args.no_warmup,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_files_per_second": round(len(result_rows) / wall_seconds, 3),
        "inference_mean_ms": round(mean(inference_times), 3),
        "inference_median_ms": round(median(inference_times), 3),
        "inference_p95_ms": round(percentile(inference_times, 0.95), 3),
        "end_to_end_mean_ms": round(mean(end_to_end_times), 3),
        "end_to_end_median_ms": round(median(end_to_end_times), 3),
        "end_to_end_p95_ms": round(percentile(end_to_end_times, 0.95), 3),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    best_metric = max(metric_rows, key=lambda row: (float(row["accuracy"]), float(row["recall"])))
    print(f"Saved per-file results: {output_path}")
    print(f"Saved threshold metrics: {metrics_output_path}")
    print(f"Saved runtime summary: {args.summary_output.resolve()}")
    print(
        "Best threshold by accuracy: "
        f"{best_metric['threshold']} "
        f"(accuracy={best_metric['accuracy']}, "
        f"fp={best_metric['fp']}, fn={best_metric['fn']})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the anti-spoofing model on datasets/anti_spoofing/real and fake.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/anti_spoofing"),
        help="Dataset directory containing real/ and fake/ subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/anti_spoofing_results.csv"),
        help="Per-file output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/anti_spoofing_runtime.json"),
        help="Runtime summary JSON path.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/anti_spoofing_threshold_metrics.csv"),
        help="Threshold sweep metrics CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated thresholds to evaluate. Example: 0.05,0.07,0.10",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to evaluate for quick smoke tests.",
    )
    parser.add_argument(
        "--single-label",
        choices=("real", "fake"),
        default=None,
        help="Assign one label to every audio file directly under --dataset.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print progress every 25 files.")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Include first-use kernel initialization instead of warming up before measurement.",
    )
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
