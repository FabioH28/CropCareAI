"""Build the production open-set crop gate artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.open_set_gate import build_open_set_gate
from ml.plant.dl.classifier.paths import PRODUCTION_MODEL_PATH, PRODUCTION_OPEN_SET_GATE_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CropCare open-set crop gate artifact.")
    parser.add_argument("--checkpoint", default=str(PRODUCTION_MODEL_PATH))
    parser.add_argument("--output", default=str(PRODUCTION_OPEN_SET_GATE_PATH))
    parser.add_argument("--samples-per-label", type=int, default=6)
    parser.add_argument("--samples-per-unknown-group", type=int, default=24)
    parser.add_argument("--minimum-supported-similarity", type=float, default=0.58)
    parser.add_argument("--unknown-margin", type=float, default=0.03)
    args = parser.parse_args()

    artifact = build_open_set_gate(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        samples_per_label=args.samples_per_label,
        samples_per_unknown_group=args.samples_per_unknown_group,
        minimum_supported_similarity=args.minimum_supported_similarity,
        unknown_margin=args.unknown_margin,
    )

    print(
        f"Saved {artifact['version']} to {Path(args.output).resolve()} "
        f"with {len(artifact['supported_prototypes'])} supported crops and "
        f"{len(artifact['unknown_prototypes'])} unknown groups."
    )


if __name__ == "__main__":
    main()
