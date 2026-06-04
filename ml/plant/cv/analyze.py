"""CLI wrapper for the plant computer-vision analysis pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.common.utils import write_json
from ml.plant.cv.analysis_pipeline import analyze_plant_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full plant CV analysis pipeline.")
    parser.add_argument("image_path", help="Path to the input plant image.")
    parser.add_argument("--output-dir", required=True, help="Directory where intermediate outputs will be saved.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the JSON analysis payload.")
    parser.add_argument("--crop-hint", default=None, help="Optional crop hint to carry into the final prediction.")
    args = parser.parse_args()

    analysis = analyze_plant_image(
        image_path=Path(args.image_path),
        output_dir=Path(args.output_dir),
        crop_hint=args.crop_hint,
    )

    if args.output_json:
        write_json(analysis, args.output_json)

    print(json.dumps(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
