"""Build deterministic, balanced DFADD calibration and test subsets.

The source parquet files are intentionally kept out of Git. Download the two
public benchmark shards from SpeechAntiSpoofingBenchmarks/DFADD, place them
under datasets/public/dfadd/raw, then run this script.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "public" / "dfadd" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "public" / "dfadd"
FAKE_PER_GENERATOR = {"calibration": 40, "test": 80}
REAL_LIMITS = {"calibration": 200, "test": 400}


def _content_group(path: str, speaker: str) -> str:
    """Group matching real/fake source utterances to prevent split leakage."""

    match = re.search(rf"{re.escape(speaker)}_(\d+)", path, re.IGNORECASE)
    utterance_number = match.group(1) if match else Path(path).stem
    return f"{speaker.lower()}_{utterance_number.lower()}"


def _split_for_group(content_group: str) -> str:
    bucket = int(sha256(content_group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "calibration" if bucket < 33 else "test"


def _stable_order(row: dict[str, object]) -> str:
    value = f"dfadd-v1:{row['utterance_id']}"
    return sha256(value.encode("utf-8")).hexdigest()


def collect_rows(source_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parquet_path in sorted(source_dir.glob("test-*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(
                row_group,
                columns=["path", "audio", "label", "notes"],
            )
            for row in table.to_pylist():
                metadata = json.loads(row["notes"])
                label = "fake" if int(row["label"]) == 1 else "real"
                generator = str(metadata.get("generator") or "unknown").lower()
                speaker = str(metadata.get("speaker") or "unknown").lower()
                utterance_id = str(metadata.get("utterance_id") or Path(row["path"]).stem)
                content_group = _content_group(str(row["path"]), speaker)
                rows.append(
                    {
                        "source_path": str(row["path"]),
                        "audio_bytes": row["audio"]["bytes"],
                        "label": label,
                        "generator": generator,
                        "speaker": speaker,
                        "utterance_id": utterance_id,
                        "content_group": content_group,
                        "split": _split_for_group(content_group),
                    }
                )
    if not rows:
        raise SystemExit(f"No test parquet shards found under {source_dir}")
    return rows


def select_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for split in ("calibration", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        real_rows = sorted(
            (row for row in split_rows if row["label"] == "real"),
            key=_stable_order,
        )
        real_limit = REAL_LIMITS[split]
        if len(real_rows) < real_limit:
            raise SystemExit(f"Not enough DFADD real rows for {split}: {len(real_rows)}")
        selected.extend(real_rows[:real_limit])

        generators = sorted(
            {
                str(row["generator"])
                for row in split_rows
                if row["label"] == "fake"
            }
        )
        if not generators:
            raise SystemExit("DFADD fake generators were not found")
        fake_limit = FAKE_PER_GENERATOR[split]
        for generator in generators:
            generator_rows = sorted(
                (
                    row
                    for row in split_rows
                    if row["label"] == "fake" and row["generator"] == generator
                ),
                key=_stable_order,
            )
            if len(generator_rows) < fake_limit:
                raise SystemExit(
                    f"Not enough DFADD {generator} rows for {split}: {len(generator_rows)}"
                )
            selected.extend(generator_rows[:fake_limit])
    return selected


def write_subset(rows: list[dict[str, object]], output_dir: Path) -> None:
    manifests: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        audio_bytes = bytes(row.pop("audio_bytes"))
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if sample_rate != 16_000:
            raise SystemExit(f"Unexpected sample rate {sample_rate}: {row['source_path']}")
        mono = audio.mean(axis=1)

        split = str(row["split"])
        label = str(row["label"])
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(row["utterance_id"]))
        destination = output_dir / split / label / f"{safe_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, mono, sample_rate, subtype="PCM_16", format="WAV")

        manifests.append(
            {
                **row,
                "file": str(destination.relative_to(output_dir)),
                "source_sha256": sha256(audio_bytes).hexdigest(),
                "duration_seconds": round(len(mono) / sample_rate, 4),
            }
        )
        if index % 100 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] extracted")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(manifests[0].keys()))
        writer.writeheader()
        writer.writerows(manifests)
    print(f"Saved manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = collect_rows(args.source.resolve())
    selected = select_rows(rows)
    write_subset(selected, args.output.resolve())

    for split in ("calibration", "test"):
        split_rows = [row for row in selected if row["split"] == split]
        real_count = sum(row["label"] == "real" for row in split_rows)
        fake_count = sum(row["label"] == "fake" for row in split_rows)
        print(f"{split}: real={real_count}, fake={fake_count}, total={len(split_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
