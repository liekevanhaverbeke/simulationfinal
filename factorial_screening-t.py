"""
Full Factorial Screening — 3 × 3 × 4 = 36 combinations
========================================================
Factors
-------
  A  urgent_slots   : 10, 14, 20          (capacity distribution)
  B  strategy       : 1, 2, 3             (timing of urgent slots)
  C  rule           : 1, 2, 3, 4          (appointment scheduling rule)

Expected CSV layout (one file per combination)
-----------------------------------------------
  output/output-S{strategy}-{slots}-R{rule}.csv

  Columns:  week, elAppWT, elScanWT, urScanWT, OT
  (produced by pipe.py / save_results)

What this script does
---------------------
  1. Reads every CSV that matches the naming pattern
  2. Applies warmup cut + batch-means (same settings as pipe.py)
  3. Derives OV = elAppWT/168 + urScanWT/9
  4. Builds the full 36-row results table with mean + 95 % CI per KPI
  5. Computes main effects and all 2-way interactions on OV
  6. Saves results + effects to CSV and plots a Pareto effects chart

Usage
-----
  python factorial_screening.py                    # looks for ./output/
  python factorial_screening.py path/to/output/   # explicit folder
"""

import sys
import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0.  Output folder (contains all the per-combination CSVs)
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR

# ---------------------------------------------------------------------------
# 1.  Factorial design
# ---------------------------------------------------------------------------
URGENT_SLOTS = list(range(10, 21))   # factor A: 10, 11, 12, ..., 20
STRATEGIES   = [1, 2, 3]            # factor B
RULES        = [1, 2, 3, 4]         # factor C

RULE_NAMES = {
    1: "FCFS",
    2: "Bailey-Welch",
    3: "Blocking",
    4: "Benchmarking",
}

RAW_KPIS = ["elAppWT", "elScanWT", "urScanWT", "OT"]
ALL_KPIS = RAW_KPIS + ["OV"]

# ---------------------------------------------------------------------------
# 2.  Batch-means settings  (must match what pipe.py used to create the CSVs)
# ---------------------------------------------------------------------------
WARMUP     = 100   # weeks to discard
BATCH_SIZE = 65    # weeks per batch

# ---------------------------------------------------------------------------
# 3.  File naming helper
# ---------------------------------------------------------------------------

def csv_path(strategy: int, slots: int, rule: int) -> str:
    """Return the expected path for one combination."""
    fname = f"output-S{strategy}-{slots}-r{rule}.csv"
    return os.path.join(OUTPUT_DIR, fname)

# ---------------------------------------------------------------------------
# 4.  Load + batch-means for one CSV
# ---------------------------------------------------------------------------

def summarise_csv(path: str) -> dict | None:
    """
    Read the weekly CSV, apply warmup cut, batch-means, return summary dict.
    Returns None if the file is missing or has insufficient data.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # drop warmup rows (positional, not by 'week' value)
    post = df.iloc[WARMUP:].copy()
    post["OV"] = post["elAppWT"] / 168.0 + post["urScanWT"] / 9.0

    n_batches = len(post) // BATCH_SIZE
    if n_batches < 2:
        return None

    usable = post.iloc[: n_batches * BATCH_SIZE]

    summary = {}
    for kpi in ALL_KPIS:
        arr  = usable[kpi].to_numpy().reshape(n_batches, BATCH_SIZE).mean(axis=1)
        mean = arr.mean()
        se   = arr.std(ddof=1) / np.sqrt(n_batches)
        hw   = stats.t.ppf(0.975, df=n_batches - 1) * se
        summary[kpi] = {
            "mean":      mean,
            "hw":        hw,
            "lower":     mean - hw,
            "upper":     mean + hw,
            "n_batches": n_batches,
        }
    return summary

# ---------------------------------------------------------------------------
# 5.  Run the full factorial
# ---------------------------------------------------------------------------

def run_full_factorial() -> pd.DataFrame:
    combos = list(itertools.product(URGENT_SLOTS, STRATEGIES, RULES))
    print(f"\n{'='*65}")
    print(f"Full factorial:  "
          f"{len(URGENT_SLOTS)} urgent-slot levels (10–20)  x  "
          f"{len(STRATEGIES)} timing strategies  x  "
          f"{len(RULES)} scheduling rules  =  {len(combos)} combinations")
    print(f"Reading CSVs from: {OUTPUT_DIR}")
    print(f"{'='*65}\n")

    records   = []
    missing   = []

    for slots, strategy, rule in combos:
        path = csv_path(strategy, slots, rule)
        tag  = f"S{strategy}-{slots:>2}-R{rule}"
        print(f"  {tag}  ({RULE_NAMES[rule]:<14}) ...", end=" ", flush=True)

        summary = summarise_csv(path)

        if summary is None:
            status = "FILE NOT FOUND" if not os.path.exists(path) else "INSUFFICIENT DATA"
            print(status)
            missing.append(os.path.basename(path))
            continue

        row = {
            "urgent_slots": slots,
            "strategy":     strategy,
            "rule":         rule,
            "rule_name":    RULE_NAMES[rule],
            "n_batches":    summary["OV"]["n_batches"],
        }
        for kpi in ALL_KPIS:
            row[f"{kpi}_mean"]  = summary[kpi]["mean"]
            row[f"{kpi}_hw"]    = summary[kpi]["hw"]
            row[f"{kpi}_lower"] = summary[kpi]["lower"]
            row[f"{kpi}_upper"] = summary[kpi]["upper"]

        records.append(row)
        print(f"OV = {summary['OV']['mean']:.5f} +/- {summary['OV']['hw']:.5f}")

    if missing:
        print(f"\n  WARNING: {len(missing)} file(s) not found:")
        for m in missing:
            print(f"    {m}")

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 6.  Per-level marginal means + contrasts vs reference level
# ---------------------------------------------------------------------------

def compute_marginal_means(df: pd.DataFrame, response: str = "OV_mean") -> dict:
    """
    For each factor compute the marginal mean of the response at every level
    (averaging over all other factors), plus the contrast vs the reference
    (lowest) level.

    Returns a dict keyed by factor name, each value being a DataFrame with
    columns: level, marginal_mean, contrast_vs_ref.
    """
    if df.empty:
        return {}

    factors = {
        "A: urgent_slots": ("urgent_slots", sorted(df["urgent_slots"].unique())),
        "B: strategy":     ("strategy",     sorted(df["strategy"].unique())),
        "C: rule":         ("rule",         sorted(df["rule"].unique())),
    }

    marginals = {}
    for label, (col, levels) in factors.items():
        rows = []
        ref_mean = df[df[col] == levels[0]][response].mean()
        for lvl in levels:
            m = df[df[col] == lvl][response].mean()
            level_label = RULE_NAMES[lvl] if col == "rule" else str(lvl)
            rows.append({
                "level":           lvl,
                "level_label":     level_label,
                "marginal_mean":   m,
                "contrast_vs_ref": m - ref_mean,
            })
        marginals[label] = pd.DataFrame(rows)

    return marginals


def compute_interaction_means(df: pd.DataFrame, response: str = "OV_mean") -> dict:
    """
    For every pair of factors compute the cell means (marginal mean for each
    combination of the two factor levels, averaging over the third factor).
    Used for interaction plots.

    Returns a dict keyed by 'AxB', 'AxC', 'BxC', each value a pivot DataFrame.
    """
    if df.empty:
        return {}

    pairs = {
        "A x B  (slots × strategy)": ("urgent_slots", "strategy"),
        "A x C  (slots × rule)":     ("urgent_slots", "rule"),
        "B x C  (strategy × rule)":  ("strategy",     "rule"),
    }

    interactions = {}
    for label, (c1, c2) in pairs.items():
        pivot = df.groupby([c1, c2])[response].mean().unstack(c2)
        # rename rule columns to names
        if c2 == "rule":
            pivot.columns = [RULE_NAMES[r] for r in pivot.columns]
        interactions[label] = pivot

    return interactions


# ---------------------------------------------------------------------------
# 7.  Pretty-print results table
# ---------------------------------------------------------------------------

def print_results(df: pd.DataFrame) -> None:
    cols = (["urgent_slots", "strategy", "rule_name", "n_batches"]
            + [f"{k}_mean" for k in ALL_KPIS]
            + [f"{k}_hw"   for k in ALL_KPIS])
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None,
                           "display.width", 240,
                           "display.float_format", "{:.5f}".format):
        print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# 8.  Plots: main-effect plots + interaction plots
# ---------------------------------------------------------------------------

def plot_main_effects(marginals: dict, out_path: str) -> None:
    """One subplot per factor showing marginal mean OV at every level."""
    n = len(marginals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, mdf) in zip(axes, marginals.items()):
        ax.plot(mdf["level_label"], mdf["marginal_mean"],
                marker="o", color="steelblue", linewidth=2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Level")
        ax.set_ylabel("Marginal mean OV")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Main-Effect Plots — Marginal Mean OV at Each Factor Level",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Main-effect plot saved  →  {out_path}")


def plot_interactions(interactions: dict, out_path: str) -> None:
    """One subplot per factor pair — lines for each level of the second factor."""
    n = len(interactions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for ax, (label, pivot) in zip(axes, interactions.items()):
        for i, col in enumerate(pivot.columns):
            ax.plot(pivot.index.astype(str), pivot[col],
                    marker="o", label=str(col),
                    color=colors[i % len(colors)], linewidth=1.8)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(pivot.index.name)
        ax.set_ylabel("Cell mean OV")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title=pivot.columns.name or "", fontsize=7, title_fontsize=8)

    fig.suptitle("Interaction Plots — Cell Mean OV for Each Factor-Pair",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Interaction plot saved  →  {out_path}")

# ---------------------------------------------------------------------------
# 9.  Save combined report CSV
# ---------------------------------------------------------------------------

def save_report(results: pd.DataFrame, marginals: dict, interactions: dict,
                best: pd.Series, plot_main_path: str, plot_int_path: str,
                out_path: str) -> None:
    """
    Write all outputs into one CSV file, separated by SECTION headers.
    Sections:
      1. Full factorial results (mean + CI per KPI for every combination)
      2. Marginal means + contrasts vs reference level, one block per factor
      3. Interaction cell means, one block per factor pair
      4. Best configuration (lowest OV mean)
      5. Plot file references
    """
    result_cols = (
        ["urgent_slots", "strategy", "rule", "rule_name", "n_batches"]
        + [f"{k}_mean"  for k in ALL_KPIS]
        + [f"{k}_hw"    for k in ALL_KPIS]
        + [f"{k}_lower" for k in ALL_KPIS]
        + [f"{k}_upper" for k in ALL_KPIS]
    )
    result_cols = [c for c in result_cols if c in results.columns]

    best_df = pd.DataFrame([{
        "urgent_slots":  int(best["urgent_slots"]),
        "strategy":      int(best["strategy"]),
        "rule":          int(best["rule"]),
        "rule_name":     best["rule_name"],
        "OV_mean":       round(best["OV_mean"],       6),
        "OV_hw":         round(best["OV_hw"],         6),
        "OV_lower":      round(best["OV_lower"],      6),
        "OV_upper":      round(best["OV_upper"],      6),
        "elAppWT_mean":  round(best["elAppWT_mean"],  6),
        "urScanWT_mean": round(best["urScanWT_mean"], 6),
        "elScanWT_mean": round(best["elScanWT_mean"], 6),
        "OT_mean":       round(best["OT_mean"],       6),
    }])

    with open(out_path, "w", newline="", encoding="utf-8") as f:

        # ── Section 1: full results ──────────────────────────────────────────
        f.write("SECTION,Factorial Results — mean and 95% CI per KPI\n")
        results[result_cols].to_csv(f, index=False, float_format="%.6f")
        f.write("\n")

        # ── Section 2: marginal means per factor ─────────────────────────────
        f.write("SECTION,Marginal Means and Contrasts vs Reference Level (OV)\n")
        for factor_label, mdf in marginals.items():
            f.write(f"factor,{factor_label}\n")
            mdf.to_csv(f, index=False, float_format="%.6f")
            f.write("\n")

        # ── Section 3: interaction cell means ───────────────────────────────
        f.write("SECTION,Interaction Cell Means (OV) — averaged over third factor\n")
        for pair_label, pivot in interactions.items():
            f.write(f"pair,{pair_label}\n")
            pivot.to_csv(f, float_format="%.6f")
            f.write("\n")

        # ── Section 4: best configuration ───────────────────────────────────
        f.write("SECTION,Best Configuration (lowest OV mean)\n")
        best_df.to_csv(f, index=False)
        f.write("\n")

        # ── Section 5: plot references ───────────────────────────────────────
        f.write("SECTION,Plots\n")
        f.write(f"main_effects_plot,{os.path.basename(plot_main_path)}\n")
        f.write(f"interaction_plot,{os.path.basename(plot_int_path)}\n")

    print(f"Combined report  →  {out_path}")


# ---------------------------------------------------------------------------
# 10.  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factorial_output")
    os.makedirs(results_dir, exist_ok=True)

    # --- run ---
    results = run_full_factorial()

    if results.empty:
        print("\nNo data loaded — check that OUTPUT_DIR contains the expected CSV files.")
        sys.exit(1)

    # --- per-level analysis ---
    marginals    = compute_marginal_means(results,    response="OV_mean")
    interactions = compute_interaction_means(results, response="OV_mean")

    # --- best configuration ---
    best = results.loc[results["OV_mean"].idxmin()]

    # --- plots ---
    plot_main_path = os.path.join(results_dir, "factorial_main_effects.png")
    plot_int_path  = os.path.join(results_dir, "factorial_interactions.png")
    plot_main_effects(marginals,    plot_main_path)
    plot_interactions(interactions, plot_int_path)

    # --- individual CSVs ---
    results_path = os.path.join(results_dir, "factorial_results.csv")
    results.to_csv(results_path, index=False, float_format="%.6f")
    print(f"Results table    →  {results_path}")

    marginals_path = os.path.join(results_dir, "factorial_marginal_means.csv")
    with open(marginals_path, "w", newline="", encoding="utf-8") as f:
        for factor_label, mdf in marginals.items():
            f.write(f"factor,{factor_label}\n")
            mdf.to_csv(f, index=False, float_format="%.6f")
            f.write("\n")
    print(f"Marginal means   →  {marginals_path}")

    # --- combined report ---
    report_path = os.path.join(results_dir, "factorial_report.csv")
    save_report(results, marginals, interactions, best,
                plot_main_path, plot_int_path, report_path)

    # --- console summary ---
    print("\n=== Factorial Results (KPI means) ===")
    print_results(results)

    print("\n=== Marginal Means & Contrasts vs Reference Level ===")
    for factor_label, mdf in marginals.items():
        print(f"\n  {factor_label}")
        print(mdf.to_string(index=False))

    print(f"\n=== Best configuration (lowest OV) ===")
    print(f"  urgent_slots = {int(best['urgent_slots'])}")
    print(f"  strategy     = {int(best['strategy'])}")
    print(f"  rule         = {int(best['rule'])}  ({best['rule_name']})")
    print(f"  OV mean      = {best['OV_mean']:.5f}  +/- {best['OV_hw']:.5f}")
    print(f"  elAppWT      = {best['elAppWT_mean']:.4f}  h")
    print(f"  urScanWT     = {best['urScanWT_mean']:.4f}  h")
    print(f"  OT           = {best['OT_mean']:.4f}  h")
