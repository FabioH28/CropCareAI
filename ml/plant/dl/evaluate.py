"""Evaluate the current plant model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.plant.dl.classifier.paths import REPORT_DIR


def load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    report_dir = REPORT_DIR
    summary_path = report_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found at {summary_path}. Run dl/train.py first.")
    summary = load_report(summary_path)
    print(json.dumps(summary, indent=2))
