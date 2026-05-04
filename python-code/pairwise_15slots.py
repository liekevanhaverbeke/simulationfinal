"""
Pairwise CI Analysis — Bonferroni-corrected comparisons within a slot
======================================================================
For a chosen slot level, analyses all 8 systems (S2/S3 × 4 rules) and
constructs Bonferroni-corrected simultaneous confidence intervals for
every pairwise difference in OV performance.

Bonferroni correction
---------------------
  k   = 8 systems
  c   = k*(k-1)/2 = 28 pairwise comparisons
  α   = 0.05  (overall)
  Individual CI level = 1 - α/c = 1 - 0.05/28 ≈ 99.821%

TO CHANGE SLOT: edit FOCUS_SLOT below.

Outputs (all saved to ./pairwise_output/)
-----------------------------------------
  pairwise_forest_slot<N>.png   — forest plot of all 28 Δ OV CIs
  pairwise_full_slot<N>.csv     — one row per pair, all stats

Simulation parameters are imported from pipe.py (warmup, batch_size,
n_batches).  Make sure pipe.py is importable (same directory or on
sys.path).

Usage
-----
  python comparison_pairwise.py                    # looks for ../output/
  python comparison_pairwise.py path/to/output/   # explicit path
"""

import sys, os, itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Import simulation parameters from pipe.py ────────────────────────────────
# pipe.py must be on the Python path (same directory works).


_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    import pipe as _pipe
    WARMUP     = 0
    BATCH_SIZE = _pipe.batch_size
    N_BATCHES  = _pipe.n_batches
    print(f"[pipe.py] warmup={WARMUP}, batch_size={BATCH_SIZE}, "
          f"n_batches={N_BATCHES}")
except ImportError:
    print("WARNING: pipe.py not importable — using fallback defaults.")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
FOCUS_SLOT = 15        # ← change to 12, 13, 14, 16, etc.

STRATEGIES  = [2, 3]
RULES       = [1, 2, 3, 4]
RULE_NAMES  = {1: "FCFS", 2: "Bailey-Welch", 3: "Blocking", 4: "Benchmarking"}
STRAT_NAMES = {2: "S2", 3: "S3"}

# Bonferroni parameters
ALPHA_OVERALL = 0.05
K_SYSTEMS     = len(STRATEGIES) * len(RULES)          # 8
C_PAIRS       = K_SYSTEMS * (K_SYSTEMS - 1) // 2      # 28
ALPHA_IND     = ALPHA_OVERALL / C_PAIRS                # 0.05/28
CI_LEVEL_IND  = 1 - ALPHA_IND                         # ≈ 0.99821

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "output")
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
PLOT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pairwise_output")

# Colours
STRAT_COLORS = {2: "#534AB7", 3: "#993C1D"}
RULE_COLORS  = {1: "#378ADD", 2: "#EF9F27", 3: "#1D9E75", 4: "#D85A30"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140,
})

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def csv_path(strategy, slots, rule):
    return os.path.join(OUTPUT_DIR,
                        f"output-S{strategy}-{slots}-rule{rule}.csv")

def summarise_csv(path):
    """
    Warmup cut → batch-means → mean + batch-mean array for OV.
    Uses WARMUP, BATCH_SIZE, N_BATCHES from pipe.py.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    post = df.iloc[WARMUP:].copy()
    post["OV"] = post["elAppWT"] / 168.0 + post["urScanWT"] / 9.0

    n_batches = len(post) // BATCH_SIZE
    if n_batches < 2:
        return None
    usable = post.iloc[: n_batches * BATCH_SIZE]
    arr = usable["OV"].to_numpy().reshape(n_batches, BATCH_SIZE).mean(axis=1)
    return {"mean": arr.mean(), "batch_means": arr, "n_batches": n_batches}

def load_systems(slot):
    """
    Load all 8 systems for the given slot.
    Returns a list of dicts with keys: label, strategy, rule, mean, batch_means.
    """
    systems = []
    for strategy, rule in itertools.product(STRATEGIES, RULES):
        path = csv_path(strategy, slot, rule)
        s    = summarise_csv(path)
        if s is None:
            print(f"  WARNING: not found — {os.path.basename(path)}")
            continue
        systems.append({
            "label":      f"S{strategy}-{RULE_NAMES[rule]}",
            "strategy":   strategy,
            "rule":       rule,
            "rule_name":  RULE_NAMES[rule],
            "mean":       s["mean"],
            "batch_means": s["batch_means"],
            "n_batches":  s["n_batches"],
        })
    return systems

# ── PAIRWISE CI COMPUTATION ──────────────────────────────────────────────────

def compute_pairwise(systems):
    """
    For every pair (i, j) with i < j, compute the Bonferroni-corrected CI
    for Δ OV = OV_i − OV_j.

    The individual t-quantile uses df = n_batches - 1 (Welch approximation
    when both sides share the same n_batches; the difference of batch means
    is itself a sample, so we use n_batches − 1 degrees of freedom on the
    paired differences).
    """
    records = []
    t_crit  = stats.t.ppf(1 - ALPHA_IND / 2, df=N_BATCHES - 1)

    for i, j in itertools.combinations(range(len(systems)), 2):
        sA, sB = systems[i], systems[j]

        # Paired differences of batch means (same replications)
        n   = min(len(sA["batch_means"]), len(sB["batch_means"]))
        diffs = sA["batch_means"][:n] - sB["batch_means"][:n]
        delta = diffs.mean()
        se    = diffs.std(ddof=1) / np.sqrt(n)
        hw    = t_crit * se
        df_pair = n - 1

        sig = abs(delta) > hw  # CI excludes zero → significant

        records.append({
            "system_A":        sA["label"],
            "system_B":        sB["label"],
            "strategy_A":      sA["strategy"],
            "strategy_B":      sB["strategy"],
            "rule_A":          sA["rule_name"],
            "rule_B":          sB["rule_name"],
            "OV_mean_A":       sA["mean"],
            "OV_mean_B":       sB["mean"],
            "delta_OV":        delta,      # positive → A worse than B
            "hw_bonferroni":   hw,
            "lower":           delta - hw,
            "upper":           delta + hw,
            "df":              df_pair,
            "ci_level_ind":    CI_LEVEL_IND,
            "significant":     sig,
            "A_better":        sig and (delta < 0),   # A has lower OV
            "B_better":        sig and (delta > 0),   # B has lower OV
        })

    return pd.DataFrame(records)

# ── FOREST PLOT ───────────────────────────────────────────────────────────────

def plot_forest(df_pairs, slot):
    df = df_pairs.copy().sort_values("delta_OV")
    n  = len(df)

    fig_height = max(8, n * 0.38)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    fig.suptitle(
        f"Pairwise OV differences — slot {slot}\n"
        f"Bonferroni-corrected {CI_LEVEL_IND*100:.3f}% individual CIs  "
        f"(overall α = {ALPHA_OVERALL}, c = {C_PAIRS} pairs)\n"
        f"[Δ OV = System A − System B;  negative → A is better]",
        fontsize=11, y=1.01)

    colors = []
    for _, row in df.iterrows():
        if row["A_better"]:    colors.append("#1D9E75")   # A significantly better
        elif row["B_better"]:  colors.append("#D85A30")   # B significantly better
        else:                  colors.append("#B4B2A9")   # not significant

    y_pos = np.arange(n)

    ax.axvline(0, color="black", lw=1.0, ls="--", zorder=1)
    ax.barh(y_pos, df["delta_OV"],
            xerr=df["hw_bonferroni"],
            color=colors, alpha=0.82,
            error_kw={"elinewidth": 1.1, "capsize": 3},
            zorder=2)

    # Y-axis labels: "A vs B"
    labels = [f"{row.system_A}  vs  {row.system_B}" for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Δ OV  (A − B)", fontsize=10)

    # Annotate delta value at end of each bar
    for i, (_, row) in enumerate(df.iterrows()):
        sign = "+" if row["delta_OV"] > 0 else ""
        x_txt = row["delta_OV"] + np.sign(row["delta_OV"]) * (row["hw_bonferroni"] * 0.08)
        ha    = "left" if row["delta_OV"] >= 0 else "right"
        ax.text(x_txt, i, f"{sign}{row['delta_OV']:.4f}",
                va="center", ha=ha, fontsize=6.8, color="#333333")

    legend_patches = [
        Patch(color="#1D9E75", alpha=0.82, label="A significantly better (lower OV)"),
        Patch(color="#D85A30", alpha=0.82, label="B significantly better (lower OV)"),
        Patch(color="#B4B2A9", alpha=0.82, label="Not statistically significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              frameon=False, fontsize=8)

    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, f"pairwise_forest_slot{slot}.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fp}")

# ── CSV OUTPUT ────────────────────────────────────────────────────────────────

def save_csv(df_pairs, slot):
    fp = os.path.join(PLOT_DIR, f"pairwise_full_slot{slot}.csv")
    df_pairs.to_csv(fp, index=False, float_format="%.6f")
    print(f"  Saved {fp}")

# ── SUMMARY PRINT ─────────────────────────────────────────────────────────────

def print_summary(df_pairs):
    n_sig   = df_pairs["significant"].sum()
    n_total = len(df_pairs)
    print(f"\n  {n_sig}/{n_total} pairs show a statistically significant difference "
          f"(Bonferroni-corrected, overall α={ALPHA_OVERALL})")

    sig = df_pairs[df_pairs["significant"]].sort_values("delta_OV")
    if sig.empty:
        print("  No significant differences found.")
    else:
        print(f"\n  Significant pairs (sorted by Δ OV):")
        print(f"  {'System A':<22} {'System B':<22} "
              f"{'Δ OV':>9} {'±HW':>9} {'Better'}")
        print("  " + "-" * 75)
        for _, row in sig.iterrows():
            better = row["system_A"] if row["A_better"] else row["system_B"]
            print(f"  {row['system_A']:<22} {row['system_B']:<22} "
                  f"{row['delta_OV']:>+9.5f} {row['hw_bonferroni']:>9.5f}  {better}")

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 65)
    print(f"Pairwise CI Analysis — slot {FOCUS_SLOT}")
    print(f"  k systems : {K_SYSTEMS}")
    print(f"  c pairs   : {C_PAIRS}")
    print(f"  α overall : {ALPHA_OVERALL}")
    print(f"  α indiv.  : {ALPHA_IND:.6f}  →  {CI_LEVEL_IND*100:.4f}% individual CIs")
    print(f"  Warmup    : {WARMUP}  |  Batch size: {BATCH_SIZE}  |  N batches: {N_BATCHES}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Plots dir : {PLOT_DIR}")
    print("=" * 65 + "\n")

    print(f"Loading {K_SYSTEMS} systems for slot {FOCUS_SLOT} ...")
    systems = load_systems(FOCUS_SLOT)
    if len(systems) < 2:
        print("Not enough systems loaded. Check OUTPUT_DIR and file naming.")
        sys.exit(1)
    print(f"  Loaded {len(systems)} systems.\n")

    print(f"Computing {C_PAIRS} pairwise Bonferroni CIs ...")
    df_pairs = compute_pairwise(systems)

    print_summary(df_pairs)

    print("\nGenerating forest plot ...")
    plot_forest(df_pairs, FOCUS_SLOT)

    print("\nSaving CSV ...")
    save_csv(df_pairs, FOCUS_SLOT)

    print(f"\nDone. All outputs in: {PLOT_DIR}/")