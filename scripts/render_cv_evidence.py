"""Render the full CV-evidence image set for one image (for demos/slides).

Runs the deployed CV pipeline (ml.plant.cv.analysis_pipeline) and writes the
numbered stage images plus analysis.json into an output folder. Use this to
regenerate clean demo images after any change to the CV pipeline.

Usage:
  python scripts/render_cv_evidence.py <image> [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.plant.cv.analysis_pipeline import analyze_plant_image  # noqa: E402

DEFAULT_OUT = ROOT / "ml/plant/artifacts/reports/cv_evidence_sample"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    image = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT / image.stem[:40]
    out.mkdir(parents=True, exist_ok=True)
    res = analyze_plant_image(image, out)
    seg = res["segmentation"]
    summary = res["summary"]
    print(f"image: {image.name}")
    print(f"  -> {out}")
    print(f"  leaf_area_%={seg['leaf_area_percentage']}  "
          f"consensus_infected_%={seg['consensus_infected_area_percentage']}  "
          f"lesion_regions={res['postprocessing']['lesion_region_count']}")
    print(f"  predicted_crop={summary['predicted_crop']}  "
          f"health={summary['health_status']}  severity={summary['severity_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
