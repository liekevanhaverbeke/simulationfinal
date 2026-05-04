"""
Comparative Design: Bonferroni comparison vs. base
=============================================================

This script reads the raw simulation output CSV files directly from the
project output folder and compares the 24 retained candidate designs with
one standard/base design.

Base / standard design:
    S1 × Rule 1 (FCFS) × 14 urgent slots

Candidate designs:
    S2/S3 × 4 rules × slots 15, 16, 17
    = 2 × 4 × 3 = 24 alternatives

Bonferroni setup:
    k = 25 total model variants = 1 base + 24 alternatives
    c = k - 1 = 24 simultaneous comparisons vs. the base
    alpha_per_CI = alpha / c = 0.05 / 24
    individual CI level = 1 - alpha/c = 99.792%

Method:
    Welch confidence intervals, based on batch means of the OV output.
    OV = elAppWT / 168 + urScanWT / 9

"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats


# ── CONFIG ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
RESULTS_DIR = SCRIPT_DIR / "bonferroni_output"

WARMUP = 100
BATCH_SIZE = 65
ALPHA = 0.05  # overall significance level, i.e. overall 95% confidence

BASE_STRATEGY = 1
BASE_RULE = 1
BASE_SLOTS = 14

CANDIDATE_STRATEGIES = [2, 3]
CANDIDATE_RULES = [1, 2, 3, 4]
CANDIDATE_SLOTS = [15, 16, 17]

RULE_NAMES = {
    1: "FCFS",
    2: "Bailey-Welch",
    3: "Blocking",
    4: "Benchmarking",
}

# Explicit Bonferroni bookkeeping.
N_CANDIDATES = len(CANDIDATE_STRATEGIES) * len(CANDIDATE_RULES) * len(CANDIDATE_SLOTS)
K_TOTAL = 1 + N_CANDIDATES
C_COMPARISONS = K_TOTAL - 1

assert K_TOTAL == 25, f"Expected k=25, got k={K_TOTAL}"
assert C_COMPARISONS == 24, f"Expected c=24, got c={C_COMPARISONS}"

COLORS = {
    "BETTER": "#1D9E75",
    "WORSE": "#D85A30",
    "n.s.": "#B4B2A9",
}


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def csv_path(strategy: int, slots: int, rule: int) -> Path:
    return OUTPUT_DIR / f"output-S{strategy}-{slots}-rule{rule}.csv"


def load_batch_ov(strategy: int, slots: int, rule: int) -> np.ndarray:
    path = csv_path(strategy, slots, rule)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Check whether OUTPUT_DIR is correct. Current OUTPUT_DIR: {OUTPUT_DIR}"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_cols = {"elAppWT", "urScanWT"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path.name} is missing required columns: {missing_cols}")

    post = df.iloc[WARMUP:].copy()
    post["OV"] = post["elAppWT"] / 168.0 + post["urScanWT"] / 9.0

    n_batches = len(post) // BATCH_SIZE
    if n_batches < 2:
        raise ValueError(
            f"Not enough post-warm-up observations in {path.name}: "
            f"{len(post)} rows after warm-up -> {n_batches} batches."
        )

    usable = post.iloc[: n_batches * BATCH_SIZE]
    batch_means = usable["OV"].to_numpy().reshape(n_batches, BATCH_SIZE).mean(axis=1)
    return batch_means


def summarise_batch_means(batch_means: np.ndarray) -> dict:
    n = len(batch_means)
    mean = float(batch_means.mean())
    var = float(batch_means.var(ddof=1))
    se = float(np.sqrt(var / n))
    return {"n_batches": n, "mean": mean, "var": var, "se": se, "df": n - 1}


# ── BONFERRONI-WELCH COMPARISON ───────────────────────────────────────────────
def welch_bonferroni_interval(candidate: np.ndarray, base: np.ndarray) -> dict:
    cand = summarise_batch_means(candidate)
    base_sum = summarise_batch_means(base)

    delta = cand["mean"] - base_sum["mean"]
    se_diff = np.sqrt(cand["var"] / cand["n_batches"] + base_sum["var"] / base_sum["n_batches"])

    numerator = (cand["var"] / cand["n_batches"] + base_sum["var"] / base_sum["n_batches"]) ** 2
    denominator = (
        ((cand["var"] / cand["n_batches"]) ** 2) / (cand["n_batches"] - 1)
        + ((base_sum["var"] / base_sum["n_batches"]) ** 2) / (base_sum["n_batches"] - 1)
    )
    welch_df = numerator / denominator if denominator > 0 else min(cand["df"], base_sum["df"])

    alpha_per_ci = ALPHA / C_COMPARISONS
    ci_level = 1 - alpha_per_ci
    t_crit = stats.t.ppf(1 - alpha_per_ci / 2, df=welch_df)
    half_width = t_crit * se_diff

    ci_lower = delta - half_width
    ci_upper = delta + half_width

    if ci_upper < 0:
        direction = "BETTER"
        interpretation = "significantly better than base"
    elif ci_lower > 0:
        direction = "WORSE"
        interpretation = "significantly worse than base"
    else:
        direction = "n.s."
        interpretation = "not significantly different from base"

    return {
        "base_mean": base_sum["mean"],
        "candidate_mean": cand["mean"],
        "delta_OV": delta,
        "se_delta": se_diff,
        "welch_df": welch_df,
        "t_crit": t_crit,
        "half_width_bonferroni": half_width,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha_overall": ALPHA,
        "alpha_per_ci": alpha_per_ci,
        "ci_level_pct": ci_level * 100,
        "n_base_batches": base_sum["n_batches"],
        "n_candidate_batches": cand["n_batches"],
        "direction": direction,
        "interpretation": interpretation,
        "significant": direction != "n.s.",
    }


def run_comparisons() -> pd.DataFrame:
    print("=" * 78)
    print("PHASE 2 — BONFERRONI COMPARISON VS. BASE")
    print("=" * 78)
    print(f"Raw output folder: {OUTPUT_DIR}")
    print(f"Results folder:    {RESULTS_DIR}")
    print(f"Base design:       S{BASE_STRATEGY} × {RULE_NAMES[BASE_RULE]} × {BASE_SLOTS} urgent slots")
    print(
        f"Candidates:        S2/S3 × 4 rules × slots 15/16/17 "
        f"= {N_CANDIDATES} alternatives"
    )
    print(f"k = {K_TOTAL} total model variants = 1 base + {N_CANDIDATES} alternatives")
    print(f"c = k - 1 = {C_COMPARISONS} simultaneous comparisons")
    print(
        f"Bonferroni:        alpha_per_CI = {ALPHA}/{C_COMPARISONS} "
        f"= {ALPHA / C_COMPARISONS:.6f}; "
        f"individual CI level = {(1 - ALPHA / C_COMPARISONS) * 100:.3f}%"
    )
    print(f"Overall confidence level >= {(1 - ALPHA) * 100:.1f}%")
    print("=" * 78)

    base_batches = load_batch_ov(BASE_STRATEGY, BASE_SLOTS, BASE_RULE)
    base_summary = summarise_batch_means(base_batches)
    print(
        f"\nBase OV mean = {base_summary['mean']:.6f} "
        f"based on {base_summary['n_batches']} batch means\n"
    )

    rows = []
    for slots in CANDIDATE_SLOTS:
        for strategy in CANDIDATE_STRATEGIES:
            for rule in CANDIDATE_RULES:
                candidate_batches = load_batch_ov(strategy, slots, rule)
                res = welch_bonferroni_interval(candidate_batches, base_batches)

                label = f"{slots} slots | S{strategy} - {RULE_NAMES[rule]}"
                rows.append({
                    "label": label,
                    "urgent_slots": slots,
                    "strategy": strategy,
                    "rule": rule,
                    "rule_name": RULE_NAMES[rule],
                    **res,
                })

    results = pd.DataFrame(rows)

    if len(results) != C_COMPARISONS:
        raise ValueError(
            f"Expected {C_COMPARISONS} comparisons, got {len(results)}. "
            "Check CANDIDATE_STRATEGIES, CANDIDATE_RULES, and CANDIDATE_SLOTS."
        )

    results = results.sort_values("delta_OV").reset_index(drop=True)
    results.insert(0, "rank_by_delta", np.arange(1, len(results) + 1))
    return results


# ── REPORTING ────────────────────────────────────────────────────────────────
def print_report(results: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("RESULTS SORTED BY DELTA OV — negative means better than base")
    print("=" * 78)

    display_cols = [
        "rank_by_delta", "label", "candidate_mean", "base_mean", "delta_OV",
        "ci_lower", "ci_upper", "interpretation"
    ]
    print(
        results[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(results["interpretation"].value_counts().to_string())

    better = results[results["direction"] == "BETTER"]
    ns = results[results["direction"] == "n.s."]
    worse = results[results["direction"] == "WORSE"]

    if not better.empty:
        print("\nSignificantly better than base:")
        for _, row in better.iterrows():
            print(
                f"  - {row['label']}: delta={row['delta_OV']:.6f}, "
                f"CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
            )
    else:
        print("\nNo candidate is significantly better than the base.")

    if not ns.empty:
        print("\nNot significantly different from base:")
        for _, row in ns.iterrows():
            print(
                f"  - {row['label']}: delta={row['delta_OV']:.6f}, "
                f"CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
            )

    if not worse.empty:
        print("\nSignificantly worse than base:")
        for _, row in worse.iterrows():
            print(
                f"  - {row['label']}: delta={row['delta_OV']:.6f}, "
                f"CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
            )


# ── PLOT ─────────────────────────────────────────────────────────────────────
def plot_forest(results: pd.DataFrame) -> Path:
    plot_df = results.sort_values("delta_OV", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))

    fig_height = max(8, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(11, fig_height))

    ax.axvline(0, color="black", linewidth=1.1, linestyle="--", zorder=1)

    colors = [COLORS[d] for d in plot_df["direction"]]
    xerr = [
        plot_df["delta_OV"] - plot_df["ci_lower"],
        plot_df["ci_upper"] - plot_df["delta_OV"],
    ]

    ax.barh(
        y_pos,
        plot_df["delta_OV"],
        xerr=xerr,
        color=colors,
        alpha=0.85,
        height=0.65,
        error_kw={"elinewidth": 1.1, "capsize": 3, "ecolor": "#444"},
        zorder=2,
    )

    for i, row in plot_df.iterrows():
        offset = 0.001 if row["delta_OV"] >= 0 else -0.001
        ha = "left" if row["delta_OV"] >= 0 else "right"
        ax.text(
            row["delta_OV"] + offset,
            i,
            f"{row['delta_OV']:+.5f}",
            va="center",
            ha=ha,
            fontsize=8,
            color="#333",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("Delta OV (candidate − base)")
    ax.set_title(
        "Bonferroni comparison vs. base\n"
        f"Base: S{BASE_STRATEGY} × {RULE_NAMES[BASE_RULE]} × {BASE_SLOTS} slots | "
        f"k={K_TOTAL}, c={C_COMPARISONS}, per-CI={(1 - ALPHA / C_COMPARISONS) * 100:.3f}%, "
        f"overall ≥ {(1 - ALPHA) * 100:.0f}%",
        fontsize=11,
        pad=12,
    )

    legend_handles = [
        Patch(color=COLORS["BETTER"], alpha=0.85, label="Significantly better than base"),
        Patch(color=COLORS["n.s."], alpha=0.85, label="Not significantly different"),
        Patch(color=COLORS["WORSE"], alpha=0.85, label="Significantly worse than base"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.9)

    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = RESULTS_DIR / "bonferroni_comparison_plot.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    return out_path


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_comparisons()
    print_report(results)

    table_path = RESULTS_DIR / "bonferroni_comparison_table.csv"
    results.to_csv(table_path, index=False, float_format="%.8f")
    print(f"\nTable saved -> {table_path}")

    plot_path = plot_forest(results)
    print(f"Plot saved  -> {plot_path}")


if __name__ == "__main__":
    main()