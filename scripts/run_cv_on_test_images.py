"""Run the deployed CV pipeline over every held-out test image.

Held-out = the PlantDoc *test* split (the model trained on PlantVillage +
PlantDoc *train*, never on these). For each image it runs the real
``analyze_plant_image`` and saves the two key overlays — the 2-of-3 consensus
mask and the marked disease spots — plus a ``summary.csv``, so you can browse
the CV evidence across the whole test set and confirm lesion detection works.

Only the two key overlays are kept per image (not all 27 stage images) to avoid
filling the disk. SAM loads once for the whole run (it is cached per process).

Usage:
  python scripts/run_cv_on_test_images.py [input_dir] [output_dir]
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.plant.cv.analysis_pipeline import analyze_plant_image  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "test_images"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "test_images_results"


def _find(outputs, key):
    for o in outputs:
        if key in o.get("file_name", ""):
            return o["file_name"]
    return None


def main() -> int:
    images = sorted(p for p in IN.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and "_cv_results" not in str(p))
    if not images:
        print(f"No images under {IN}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} held-out test images from {IN}")

    rows, errors, zero = [], 0, 0
    for i, img in enumerate(images, 1):
        rel = img.relative_to(IN)
        dest = OUT / rel.parent
        dest.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp())
        try:
            res = analyze_plant_image(img, tmp)
            s = res["summary"]
            outs = res["outputs"]
            for key, suffix in (("consensus-overlay", "consensus"), ("marked-disease-spots", "marked")):
                fn = _find(outs, key)
                if fn and (tmp / fn).exists():
                    shutil.copyfile(tmp / fn, dest / f"{img.stem[:50]}__{suffix}.png")
            regions = res["postprocessing"]["lesion_region_count"]
            zero += int(regions == 0)
            rows.append({
                "image": str(rel), "class": rel.parent.name,
                "predicted_crop": s["predicted_crop"], "health_status": s["health_status"],
                "infected_area_pct": s["infected_area_percentage"], "lesion_regions": regions,
                "severity": s["severity_level"], "error": "",
            })
        except Exception as e:  # never let one bad image stop the batch
            errors += 1
            rows.append({
                "image": str(rel), "class": rel.parent.name, "predicted_crop": "",
                "health_status": "", "infected_area_pct": "", "lesion_regions": "",
                "severity": "", "error": repr(e)[:200],
            })
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if i % 20 == 0:
            print(f"  {i}/{len(images)} processed...", flush=True)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = len(rows) - errors
    diseased = sum(1 for r in rows if r["health_status"] in ("diseased", "suspicious"))
    print(f"\nDONE: {ok}/{len(rows)} processed OK, {errors} errors, "
          f"{zero} with 0 detected lesions, {diseased} flagged diseased/suspicious.")
    print(f"Overlays + summary.csv written to: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
