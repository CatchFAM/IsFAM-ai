"""Package the complementary clean-data and telephone-robust spectral models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1",
        type=Path,
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_v1.npz",
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_v2.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "app" / "assets" / "spectral_anti_spoof_ensemble_v2.npz",
    )
    args = parser.parse_args()

    v1_path = args.v1.resolve()
    v2_path = args.v2.resolve()
    v1 = np.load(v1_path, allow_pickle=False)
    v2 = np.load(v2_path, allow_pickle=False)
    v1_metadata = json.loads(v1_path.with_suffix(".json").read_text(encoding="utf-8"))
    v2_metadata = json.loads(v2_path.with_suffix(".json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for prefix, artifact in (("v1", v1), ("v2", v2)):
        for key in (
            "feature_mean",
            "feature_std",
            "hidden_weight",
            "hidden_bias",
            "output_weight",
            "output_bias",
        ):
            arrays[f"{prefix}_{key}"] = artifact[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {
        "model_name": "isfam/spectral-ensemble-telephone-v2",
        "feature_version": "spectral_ensemble_v2",
        "feature_count": 196,
        "threshold": 0.5,
        "combination": (
            "telephone-robust v2 controls automatic spoof decisions; "
            "v1-only alerts map to the additional-confirmation band"
        ),
        "components": [
            {
                "prefix": "v1",
                "model_name": v1_metadata["model_name"],
                "feature_version": v1_metadata.get("feature_version", "spectral_v1"),
                "threshold": v1_metadata["threshold"],
            },
            {
                "prefix": "v2",
                "model_name": v2_metadata["model_name"],
                "feature_version": v2_metadata["feature_version"],
                "threshold": v2_metadata["threshold"],
            },
        ],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
