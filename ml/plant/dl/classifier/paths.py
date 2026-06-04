"""Path helpers for local datasets and outputs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASETS_ROOT = PROJECT_ROOT / "datasets" / "plant"
PLANT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PLANT_ROOT / "artifacts"
MANIFEST_PATH = OUTPUT_ROOT / "dataset_manifest.csv"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
REPORT_DIR = OUTPUT_ROOT / "reports"
PRODUCTION_DIR = OUTPUT_ROOT / "production"
ROI_CACHE_ROOT = OUTPUT_ROOT / "roi_cache"
PRODUCTION_MODEL_PATH = PRODUCTION_DIR / "best_model.pt"
PRODUCTION_METADATA_PATH = PRODUCTION_DIR / "selected_model.json"
PRODUCTION_OPEN_SET_GATE_PATH = PRODUCTION_DIR / "open_set_gate.pt"
EXPERIMENT_ROOT = PROJECT_ROOT / "ml" / "experiments" / "plant"

PLANTVILLAGE_ROOT = DATASETS_ROOT / "plantvillage" / "Plant_leave_diseases_dataset_without_augmentation"
PLANTDOC_ROOT = DATASETS_ROOT / "plantdoc"
MENDELEY_ROOT = DATASETS_ROOT / "Dataset of Tomato Leaves"
PP2021_ROOT = DATASETS_ROOT / "plant-pathology-2021-fgvc8"
PP2020_ROOT = DATASETS_ROOT / "plant-pathology-2020-fgvc7"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
