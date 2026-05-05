"""
Tie-breaking analysis after all-pairwise comparison
===================================================

This script evaluates the 7 retained alternatives using:
- OV as the primary objective
- elScanWT and OT as secondary tie-breaking objectives

Important:
    This script is descriptive. It does not run Welch or paired pairwise tests.
    It summarises each retained design separately using ordinary 95% CIs based
    on batch means. Therefore, no Welch adjustment is needed here.

It reads the raw output CSVs from ../output/
and writes the results to:
    tie_breaking_output/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
RESULTS_DIR = SCRIPT_DIR / "tie_breaking_output"

# The output CSVs created by pipe.py already contain the post-warm-up weekly values.
# Therefore WARMUP is 0 here.
WARMUP = 0
BATCH_SIZE = 65

RULE_NAMES = {
    1: "FCFS",
    2: "Bailey-Welch",
    3: "Blocking",
    4: "Benchmarking",
}

# The 7 retained alternatives from the Bonferroni comparison vs. base
# Tuple order: (urgent_slots, strategy, rule)
RETAINED_DESIGNS = [
    (15, 3, 2),  # S3 - Bailey-Welch
    (15, 3, 4),  # S3 - Benchmarking
    (15, 3, 1),  # S3 - FCFS
    (15, 3, 3),  # S3 - Blocking
    (15, 2, 2),  # S2 - Bailey-Welch
    (15, 2, 4),  # S2 - Benchmarking
    (15, 2, 1),  # S2 - FCFS
]

GENERATED_OUTPUT_FILES = [
    "tie_breaking_secondary_objectives.csv",
    "tie_breaking_secondary_metrics.png",
]


def clean_previous_outputs() -> None:
    """Remove previous generated tie-breaking files from RESULTS_DIR."""
    for filename in GENERATED_OUTPUT_FILES:
        path = RESULTS_DIR / filename
        if path.exists():
            path.unlink()


def csv_path(strategy: int, slots: int, rule: int) -> Path:
    return OUTPUT_DIR / f"output-S{strategy}-{slots}-rule{rule}.csv"


def design_label(slots: int, strategy: int, rule: int) -> str:
    return f"{slots} slots | S{strategy} - {RULE_NAMES[rule]}"


def load_metrics(strategy: int, slots: int, rule: int) -> pd.DataFrame:
    path = csv_path(strategy, slots, rule)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = {"elAppWT", "elScanWT", "urScanWT", "OT"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    post = df.iloc[WARMUP:].copy()
    post["OV"] = post["elAppWT"] / 168.0 + post["urScanWT"] / 9.0

    n_batches = len(post) // BATCH_SIZE
    if n_batches < 2:
        raise ValueError(
            f"Not enough post-warm-up observations in {path.name}: "
            f"{len(post)} rows -> {n_batches} batches."
        )

    usable = post.iloc[: n_batches * BATCH_SIZE].copy()

    batch_rows = []
    for batch_id in range(n_batches):
        batch = usable.iloc[batch_id * BATCH_SIZE:(batch_id + 1) * BATCH_SIZE]

        batch_rows.append({
            "batch": batch_id + 1,
            "OV": batch["OV"].mean(),
            "elAppWT": batch["elAppWT"].mean(),
            "urScanWT": batch["urScanWT"].mean(),
            "elScanWT": batch["elScanWT"].mean(),
            "OT": batch["OT"].mean(),
        })

    return pd.DataFrame(batch_rows)


def mean_ci(series: pd.Series) -> dict:
    n = len(series)
    df = n - 1
    mean = series.mean()
    se = series.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=df)
    hw = tcrit * se

    return {
        "mean": mean,
        "lower95": mean - hw,
        "upper95": mean + hw,
        "hw95": hw,
        "n_batches": n,
        "df": df,
    }


def run_analysis() -> pd.DataFrame:
    rows = []

    for slots, strategy, rule in RETAINED_DESIGNS:
        label = design_label(slots, strategy, rule)
        batch_df = load_metrics(strategy, slots, rule)

        ov = mean_ci(batch_df["OV"])
        el_app = mean_ci(batch_df["elAppWT"])
        ur_scan = mean_ci(batch_df["urScanWT"])
        el_scan = mean_ci(batch_df["elScanWT"])
        ot = mean_ci(batch_df["OT"])

        rows.append({
            "label": label,
            "urgent_slots": slots,
            "strategy": strategy,
            "rule": rule,
            "rule_name": RULE_NAMES[rule],

            "OV_mean": ov["mean"],
            "OV_lower95": ov["lower95"],
            "OV_upper95": ov["upper95"],
            "OV_hw95": ov["hw95"],

            "elAppWT_mean": el_app["mean"],
            "urScanWT_mean": ur_scan["mean"],

            "elScanWT_mean": el_scan["mean"],
            "elScanWT_lower95": el_scan["lower95"],
            "elScanWT_upper95": el_scan["upper95"],
            "elScanWT_hw95": el_scan["hw95"],

            "OT_mean": ot["mean"],
            "OT_lower95": ot["lower95"],
            "OT_upper95": ot["upper95"],
            "OT_hw95": ot["hw95"],

            "n_batches": ov["n_batches"],
            "df": ov["df"],
        })

    result = pd.DataFrame(rows)

    # Ranking: primary OV, then secondary criteria.
    result["rank_OV"] = result["OV_mean"].rank(method="min", ascending=True).astype(int)
    result["rank_elScanWT"] = result["elScanWT_mean"].rank(method="min", ascending=True).astype(int)
    result["rank_OT"] = result["OT_mean"].rank(method="min", ascending=True).astype(int)

    # OV remains primary; elScanWT and OT are only used to support interpretation.
    result = result.sort_values(
        ["OV_mean", "elScanWT_mean", "OT_mean"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    result.insert(0, "final_rank", np.arange(1, len(result) + 1))

    return result


def plot_secondary_metrics(result: pd.DataFrame) -> Path:
    plot_df = result.sort_values("OV_mean", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    metrics = [
        ("OV_mean", "Mean OV"),
        ("elScanWT_mean", "Mean elective scan waiting time"),
        ("OT_mean", "Mean overtime"),
    ]

    y_pos = np.arange(len(plot_df))

    for ax, (col, title) in zip(axes, metrics):
        ax.barh(y_pos, plot_df[col], alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel(col.replace("_mean", ""))

        for i, value in enumerate(plot_df[col]):
            ax.text(value, i, f" {value:.4f}", va="center", fontsize=8)

        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["label"], fontsize=9)
    axes[0].invert_yaxis()

    plt.suptitle("Tie-breaking analysis for the 7 retained alternatives", fontsize=12)
    plt.tight_layout()

    out_path = RESULTS_DIR / "tie_breaking_secondary_metrics.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs()

    result = run_analysis()

    table_path = RESULTS_DIR / "tie_breaking_secondary_objectives.csv"
    result.to_csv(table_path, index=False, float_format="%.8f")

    plot_path = plot_secondary_metrics(result)

    print("=" * 90)
    print("TIE-BREAKING ANALYSIS")
    print("=" * 90)
    print(f"Raw output folder: {OUTPUT_DIR}")
    print(f"Results folder:    {RESULTS_DIR}")
    print("Method: descriptive batch-mean summaries; ordinary 95% CIs; df = n_batches - 1")
    print()
    print(result[
        [
            "final_rank",
            "label",
            "OV_mean",
            "elScanWT_mean",
            "OT_mean",
            "rank_OV",
            "rank_elScanWT",
            "rank_OT",
            "n_batches",
            "df",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nTable saved -> {table_path}")
    print(f"Plot saved  -> {plot_path}")


if __name__ == "__main__":
    main()
