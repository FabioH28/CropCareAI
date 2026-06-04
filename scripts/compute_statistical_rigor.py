"""Statistical-rigor add-on for the CropCare field results.

Turns the point-estimate accuracies in the cross-crop reports into defensible
statistics, all from the JSON already saved under artifacts/reports/ (no GPU,
no re-inference):

  1. Wilson 95% score intervals for every per-crop and overall accuracy
     (cascade and unified-only). Wilson is used instead of the normal
     approximation because the per-crop n is small (16-69).
  2. A bootstrap 95% CI for the cascade overall accuracy as an independent
     cross-check (uses the per-image correctness flags).
  3. Calibration of the deployed system's confidence: Expected Calibration
     Error (ECE) + a reliability diagram, from the per-image (confidence,
     correct) pairs in the cascade report.

Outputs:
  - ml/plant/artifacts/reports/statistical_rigor/statistical_rigor.json
  - ml/plant/artifacts/reports/statistical_rigor/confidence_intervals.md
  - ml/plant/artifacts/reports/figures/cross_crop_accuracy_ci.png
  - ml/plant/artifacts/reports/figures/reliability_diagram.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "ml/plant/artifacts/reports"
FIGURES = REPORTS / "figures"
OUT_DIR = REPORTS / "statistical_rigor"
CASCADE = REPORTS / "cross_crop_cascade_eval.json"
UNIFIED = REPORTS / "cross_crop_unified_only_eval.json"

CROP_ORDER = ["grape", "pepper", "apple", "corn", "tomato", "potato"]
Z95 = 1.959964  # standard normal quantile for a two-sided 95% interval


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point, lower, upper) as proportions."""
    if trials == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    return p_hat, max(0.0, center - half), min(1.0, center + half)


def bootstrap_overall(correct_flags: list[bool], resamples: int = 20000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct_flags, dtype=np.float64)
    n = arr.size
    means = arr[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def expected_calibration_error(confidences, correct, n_bins: int = 10):
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = confidences.size
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "accuracy": None, "confidence": None})
            continue
        acc = float(correct[in_bin].mean())
        conf = float(confidences[in_bin].mean())
        ece += (count / total) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "count": count, "accuracy": acc, "confidence": conf})
    return float(ece), bins


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    cascade = json.loads(CASCADE.read_text(encoding="utf-8"))
    unified = json.loads(UNIFIED.read_text(encoding="utf-8"))

    rows = []
    for crop in CROP_ORDER:
        c_tot = cascade["per_crop_total"][crop]
        c_cor = cascade["per_crop_correct"][crop]
        u_tot = unified["per_crop_total"][crop]
        u_cor = unified["per_crop_correct"][crop]
        c_p, c_lo, c_hi = wilson_interval(c_cor, c_tot)
        u_p, u_lo, u_hi = wilson_interval(u_cor, u_tot)
        rows.append(
            {
                "crop": crop, "n": c_tot,
                "unified": {"correct": u_cor, "acc": u_p, "ci95": [u_lo, u_hi]},
                "cascade": {"correct": c_cor, "acc": c_p, "ci95": [c_lo, c_hi]},
            }
        )

    c_p, c_lo, c_hi = wilson_interval(cascade["correct"], cascade["total"])
    u_p, u_lo, u_hi = wilson_interval(unified["correct"], unified["total"])
    overall = {
        "n": cascade["total"],
        "unified": {"correct": unified["correct"], "acc": u_p, "ci95_wilson": [u_lo, u_hi]},
        "cascade": {"correct": cascade["correct"], "acc": c_p, "ci95_wilson": [c_lo, c_hi]},
    }

    # Bootstrap cross-check for the cascade overall (per-image flags available).
    per_image = cascade.get("per_image", [])
    if per_image:
        flags = [bool(r["correct"]) for r in per_image]
        b_lo, b_hi = bootstrap_overall(flags)
        overall["cascade"]["ci95_bootstrap"] = [b_lo, b_hi]

    # Calibration from the deployed system's per-image confidence.
    ece, bins = (None, [])
    if per_image:
        confs = [float(r["confidence"]) for r in per_image]
        corrects = [bool(r["correct"]) for r in per_image]
        ece, bins = expected_calibration_error(confs, corrects, n_bins=10)

    summary = {
        "source_reports": {"cascade": CASCADE.name, "unified_only": UNIFIED.name},
        "method": "Wilson score 95% CI (small-n appropriate); bootstrap 95% CI cross-check; ECE 10-bin calibration",
        "per_crop": rows,
        "overall": overall,
        "calibration": {"ece_10bin": ece, "bins": bins} if ece is not None else None,
    }
    (OUT_DIR / "statistical_rigor.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- Markdown table for the README -------------------------------------
    md = []
    md.append("### Field accuracy with 95% confidence intervals\n")
    md.append("Wilson score intervals (appropriate for the small per-crop n). "
              "Point estimates match the cross-crop reports; the intervals quantify the uncertainty "
              "the small field sets carry.\n")
    md.append("| Crop | n | Unified only (95% CI) | Cascade (95% CI) |")
    md.append("|---|---|---|---|")
    for r in rows:
        u = r["unified"]; c = r["cascade"]
        md.append(
            f"| {r['crop'].capitalize()} | {r['n']} | "
            f"{fmt_pct(u['acc'])} [{fmt_pct(u['ci95'][0])}–{fmt_pct(u['ci95'][1])}] | "
            f"{fmt_pct(c['acc'])} [{fmt_pct(c['ci95'][0])}–{fmt_pct(c['ci95'][1])}] |"
        )
    o = overall
    boot = o["cascade"].get("ci95_bootstrap")
    boot_txt = f" (bootstrap [{fmt_pct(boot[0])}–{fmt_pct(boot[1])}])" if boot else ""
    md.append(
        f"| **Overall** | {o['n']} | "
        f"{fmt_pct(o['unified']['acc'])} [{fmt_pct(o['unified']['ci95_wilson'][0])}–{fmt_pct(o['unified']['ci95_wilson'][1])}] | "
        f"**{fmt_pct(o['cascade']['acc'])}** [{fmt_pct(o['cascade']['ci95_wilson'][0])}–{fmt_pct(o['cascade']['ci95_wilson'][1])}]{boot_txt} |"
    )
    if ece is not None:
        md.append(f"\n**Calibration:** Expected Calibration Error (10-bin) = **{ece:.3f}** "
                  f"on the {o['n']} field images (deployed-system confidence vs. empirical accuracy).\n")
    (OUT_DIR / "confidence_intervals.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- Figure 1: per-crop accuracy with CIs (cascade vs unified) ---------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(rows))
    c_acc = [r["cascade"]["acc"] for r in rows]
    c_err = [[r["cascade"]["acc"] - r["cascade"]["ci95"][0] for r in rows],
             [r["cascade"]["ci95"][1] - r["cascade"]["acc"] for r in rows]]
    u_acc = [r["unified"]["acc"] for r in rows]
    u_err = [[r["unified"]["acc"] - r["unified"]["ci95"][0] for r in rows],
             [r["unified"]["ci95"][1] - r["unified"]["acc"] for r in rows]]
    ax.errorbar(c_acc, y + 0.12, xerr=c_err, fmt="o", color="#1b9e3e", capsize=4,
                label="Cascade (deployed)", markersize=7)
    ax.errorbar(u_acc, y - 0.12, xerr=u_err, fmt="s", color="#888888", capsize=4,
                label="Unified only", markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['crop'].capitalize()} (n={r['n']})" for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Field accuracy (PlantDoc)  —  bars = 95% Wilson CI")
    ax.set_title("Per-crop field accuracy with 95% confidence intervals")
    ax.axvline(overall["cascade"]["acc"], color="#1b9e3e", ls="--", lw=1, alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_crop_accuracy_ci.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: reliability diagram ------------------------------------
    if ece is not None:
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        accs = [b["accuracy"] if b["accuracy"] is not None else np.nan for b in bins]
        counts = [b["count"] for b in bins]
        ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
        ax.bar(centers, accs, width=0.09, color="#1b69c4", alpha=0.8, edgecolor="white",
               label="empirical accuracy")
        for cx, a, n in zip(centers, accs, counts):
            if n > 0:
                ax.text(cx, (0 if np.isnan(a) else a) + 0.02, str(n), ha="center", fontsize=7, color="#333")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Reliability diagram (field)\nECE = {ece:.3f}, n = {overall['n']}  (bar labels = #images)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(FIGURES / "reliability_diagram.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print("Overall cascade:", fmt_pct(overall["cascade"]["acc"]),
          "Wilson", [round(x, 4) for x in overall["cascade"]["ci95_wilson"]],
          ("bootstrap " + str([round(x, 4) for x in boot])) if boot else "")
    if ece is not None:
        print(f"ECE (10-bin) = {ece:.4f}")
    print("Wrote:", OUT_DIR, "and figures in", FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
