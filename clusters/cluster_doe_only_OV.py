"""
Phase 1 — Screening Analysis (OV only)
=======================================
Produces three sets of plots focused entirely on OV:
  1a. OV trend lines by slot, faceted by strategy (lines = rule)
  1b. OV trend lines by slot, faceted by rule     (lines = strategy)
  1c. OV cluster bar charts (mean per strategy x rule per cluster)
  2.  OV ranking dot-plot per cluster
  3.  OV forest plots: delta vs. base (S1, 14 slots)

Plus CSV outputs:
  phase1_aggregated_results.csv   — full 132-row table (OV only)
  step2_rankings_by_cluster.csv   — ranked table per cluster
  step3_contrasts_vs_base.csv     — delta + CI + significance flag

Usage
-----
  python phase1_screening.py                    # looks for ../output/
  python phase1_screening.py path/to/output/   # explicit folder
"""

import sys, os, itertools
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── SETTINGS ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "output")
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR

URGENT_SLOTS = list(range(10, 21))
STRATEGIES   = [1, 2, 3]
RULES        = [1, 2, 3, 4]
RULE_NAMES   = {1: "FCFS", 2: "Bailey-Welch", 3: "Blocking", 4: "Benchmarking"}
STRAT_NAMES  = {1: "Strategy 1", 2: "Strategy 2", 3: "Strategy 3"}

WARMUP     = 100
BATCH_SIZE = 65

BASE_STRATEGY = 1
BASE_SLOTS    = 14

CLUSTER_BOUNDS = [("Low (10-14)",    10, 14),
                  ("Medium (15-17)", 15, 17),
                  ("High (18-20)",   18, 20)]
CLUSTER_ORDER  = [c[0] for c in CLUSTER_BOUNDS]

def slot_cluster(s):
    for name, lo, hi in CLUSTER_BOUNDS:
        if lo <= s <= hi:
            return name
    return "Unknown"

CLUSTER_COLORS = {"Low (10-14)":    "#1D9E75",
                  "Medium (15-17)": "#EF9F27",
                  "High (18-20)":   "#D85A30"}
RULE_COLORS    = {1: "#378ADD", 2: "#EF9F27", 3: "#1D9E75", 4: "#D85A30"}
STRAT_COLORS   = {1: "#534AB7", 2: "#0F6E56", 3: "#993C1D"}
STRAT_MARKERS  = {1: "o", 2: "s", 3: "^"}

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_outputOV")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140,
})

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def csv_path(strategy, slots, rule):
    return os.path.join(OUTPUT_DIR, f"output-S{strategy}-{slots}-rule{rule}.csv")

def summarise_csv(path):
    """Warmup cut -> batch-means -> mean + 95% CI for OV only."""
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
    arr  = usable["OV"].to_numpy().reshape(n_batches, BATCH_SIZE).mean(axis=1)
    mean = arr.mean()
    se   = arr.std(ddof=1) / np.sqrt(n_batches)
    hw   = stats.t.ppf(0.975, df=n_batches - 1) * se
    return {"mean": mean, "hw": hw,
            "lower": mean - hw, "upper": mean + hw,
            "n_batches": n_batches}

def load_all():
    combos  = list(itertools.product(URGENT_SLOTS, STRATEGIES, RULES))
    records = []
    missing = []
    print(f"Loading {len(combos)} combinations from: {OUTPUT_DIR}\n")
    for slots, strategy, rule in combos:
        path    = csv_path(strategy, slots, rule)
        summary = summarise_csv(path)
        if summary is None:
            missing.append(os.path.basename(path))
            continue
        records.append({
            "urgent_slots":   slots,
            "strategy":       strategy,
            "rule":           rule,
            "rule_name":      RULE_NAMES[rule],
            "strat_label":    STRAT_NAMES[strategy],
            "slot_cluster":   slot_cluster(slots),
            "n_batches":      summary["n_batches"],
            "OV_mean":        summary["mean"],
            "OV_hw":          summary["hw"],
            "OV_lower":       summary["lower"],
            "OV_upper":       summary["upper"],
        })

    if missing:
        print(f"  WARNING: {len(missing)} file(s) not found:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")

    df = pd.DataFrame(records)
    df["slot_cluster_ord"] = pd.Categorical(
        df["slot_cluster"], categories=CLUSTER_ORDER, ordered=True)
    print(f"Loaded {len(df)} combinations.\n")
    return df

# ── HELPERS ───────────────────────────────────────────────────────────────────

def add_cluster_bands(ax, slots):
    for name, lo, hi in CLUSTER_BOUNDS:
        present = [s for s in slots if lo <= s <= hi]
        if present:
            ax.axvspan(min(present) - 0.5, max(present) + 0.5,
                       alpha=0.08, color=CLUSTER_COLORS[name], zorder=0)

def cluster_patches():
    return [Patch(color=CLUSTER_COLORS[c], alpha=0.45, label=c)
            for c in CLUSTER_ORDER]

# ── STEP 1a — OV trend, facet by strategy, lines = rule ──────────────────────

def plot_1a(df):
    slots_sorted = sorted(df["urgent_slots"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle("OV by urgent slots — lines = scheduling rule", fontsize=12, y=1.01)

    for ax, strat_id in zip(axes, STRATEGIES):
        sub = df[df["strategy"] == strat_id].sort_values("urgent_slots")
        add_cluster_bands(ax, slots_sorted)
        for rule_id in RULES:
            r = sub[sub["rule"] == rule_id]
            if r.empty:
                continue
            x, y, hw = r["urgent_slots"].values, r["OV_mean"].values, r["OV_hw"].values
            ax.plot(x, y, marker="o", ms=4, lw=1.8,
                    color=RULE_COLORS[rule_id], label=RULE_NAMES[rule_id], zorder=3)
            ax.fill_between(x, y - hw, y + hw, alpha=0.12, color=RULE_COLORS[rule_id])
        ax.set_title(STRAT_NAMES[strat_id], fontsize=10)
        ax.set_xlabel("urgent slots")
        ax.set_ylabel("OV mean" if ax is axes[0] else "")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    rule_handles = [plt.Line2D([0], [0], color=RULE_COLORS[r], marker="o",
                               ms=4, lw=1.5, label=RULE_NAMES[r]) for r in RULES]
    fig.legend(handles=cluster_patches() + rule_handles,
               loc="lower center", ncol=7,
               bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "1a_OV_by_slot_facet_strategy.png")
    plt.savefig(fp, bbox_inches="tight"); plt.close()
    print(f"  Saved {fp}")

# ── STEP 1b — OV trend, facet by rule, lines = strategy ──────────────────────

def plot_1b(df):
    slots_sorted = sorted(df["urgent_slots"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    fig.suptitle("OV by urgent slots — lines = strategy", fontsize=12, y=1.01)

    for ax, rule_id in zip(axes, RULES):
        sub = df[df["rule"] == rule_id].sort_values("urgent_slots")
        add_cluster_bands(ax, slots_sorted)
        for strat_id in STRATEGIES:
            r = sub[sub["strategy"] == strat_id]
            if r.empty:
                continue
            x, y, hw = r["urgent_slots"].values, r["OV_mean"].values, r["OV_hw"].values
            ax.plot(x, y, marker=STRAT_MARKERS[strat_id], ms=4, lw=1.8,
                    color=STRAT_COLORS[strat_id], label=STRAT_NAMES[strat_id], zorder=3)
            ax.fill_between(x, y - hw, y + hw, alpha=0.10, color=STRAT_COLORS[strat_id])
        ax.set_title(RULE_NAMES[rule_id], fontsize=10)
        ax.set_xlabel("urgent slots")
        ax.set_ylabel("OV mean" if ax is axes[0] else "")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    strat_handles = [plt.Line2D([0], [0], color=STRAT_COLORS[s],
                                marker=STRAT_MARKERS[s], ms=4, lw=1.5,
                                label=STRAT_NAMES[s]) for s in STRATEGIES]
    fig.legend(handles=cluster_patches() + strat_handles,
               loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "1b_OV_by_slot_facet_rule.png")
    plt.savefig(fp, bbox_inches="tight"); plt.close()
    print(f"  Saved {fp}")

# ── STEP 1c — OV cluster bar charts ──────────────────────────────────────────

def plot_1c(df):
    agg = (df.groupby(["slot_cluster_ord", "strategy", "rule"], observed=True)["OV_mean"]
             .mean().reset_index().rename(columns={"OV_mean": "cell_mean"}))
    agg["label"] = agg.apply(
        lambda r: f"S{int(r['strategy'])}-{RULE_NAMES[int(r['rule'])]}", axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("OV — mean per strategy x rule, by slot cluster", fontsize=12)

    for ax, cluster in zip(axes, CLUSTER_ORDER):
        sub = agg[agg["slot_cluster_ord"] == cluster].sort_values("cell_mean")
        colors = [RULE_COLORS[int(r)] for r in sub["rule"]]
        bars = ax.barh(sub["label"], sub["cell_mean"], color=colors, alpha=0.85)
        for bar, val in zip(bars, sub["cell_mean"]):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=7)
        ax.set_title(cluster, fontsize=11, color=CLUSTER_COLORS[cluster])
        ax.set_xlabel("OV mean")

    rule_patches = [Patch(color=RULE_COLORS[r], label=RULE_NAMES[r]) for r in RULES]
    fig.legend(handles=rule_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False, title="Rule", fontsize=8)
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "1c_OV_cluster_bars.png")
    plt.savefig(fp, bbox_inches="tight"); plt.close()
    print(f"  Saved {fp}")

# ── STEP 2 — Rank combinations within each cluster ───────────────────────────

def rank_combinations(df):
    ranking_rows = []
    for cluster in CLUSTER_ORDER:
        sub  = df[df["slot_cluster"] == cluster]
        cell = (sub.groupby(["strategy", "rule"], observed=True)["OV_mean"]
                   .mean().reset_index())
        cell["slot_cluster"] = cluster
        cell = cell.sort_values("OV_mean").reset_index(drop=True)
        cell["rank"]     = cell.index + 1
        cell["label"]    = cell.apply(
            lambda r: f"S{int(r['strategy'])} - {RULE_NAMES[int(r['rule'])]}", axis=1)
        cell["is_best"]  = cell["rank"] == 1
        cell["is_worst"] = cell["rank"] == len(cell)
        ranking_rows.append(cell)

    rankings = pd.concat(ranking_rows, ignore_index=True)
    fp = os.path.join(PLOT_DIR, "step2_rankings_by_cluster.csv")
    rankings.to_csv(fp, index=False, float_format="%.6f")
    print(f"  Rankings saved -> {fp}")

    for cluster in CLUSTER_ORDER:
        sub = rankings[rankings["slot_cluster"] == cluster]
        print(f"\n  -- {cluster} (sorted by OV, best first) --")
        print(sub[["rank", "label", "OV_mean"]].to_string(
              index=False, float_format=lambda x: f"{x:.5f}"))

    # Dot-plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("OV ranking per cluster — strategy x rule combinations", fontsize=12)

    for ax, cluster in zip(axes, CLUSTER_ORDER):
        sub = rankings[rankings["slot_cluster"] == cluster].sort_values("rank", ascending=False)
        y = range(len(sub))
        dot_colors = ["#1D9E75" if r else ("#D85A30" if w else "#888780")
                      for r, w in zip(sub["is_best"], sub["is_worst"])]
        ax.scatter(sub["OV_mean"], list(y), c=dot_colors, s=90, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.set_title(cluster, fontsize=11, color=CLUSTER_COLORS[cluster])
        ax.set_xlabel("OV mean")

    fig.legend(handles=[Patch(color="#1D9E75", label="Best"),
                        Patch(color="#D85A30", label="Worst"),
                        Patch(color="#888780", label="Other")],
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "step2_OV_ranking_dotplot.png")
    plt.savefig(fp, bbox_inches="tight"); plt.close()
    print(f"  Saved {fp}")
    return rankings

# ── STEP 3 — Contrast vs base ─────────────────────────────────────────────────

def contrast_vs_base(df):
    base_rows = df[(df["strategy"] == BASE_STRATEGY) & (df["urgent_slots"] == BASE_SLOTS)]
    if base_rows.empty:
        print(f"  ERROR: base not found (S{BASE_STRATEGY}, slots={BASE_SLOTS})")
        return pd.DataFrame()

    base_mean = base_rows["OV_mean"].mean()
    base_hw   = base_rows["OV_hw"].mean()
    print(f"\n  Base OV (S{BASE_STRATEGY}, {BASE_SLOTS} slots, avg over rules): "
          f"{base_mean:.5f} +/- {base_hw:.5f}")

    rows = []
    for _, row in df.iterrows():
        delta    = row["OV_mean"] - base_mean
        hw_delta = np.sqrt(row["OV_hw"]**2 + base_hw**2)
        rows.append({
            "urgent_slots":  row["urgent_slots"],
            "strategy":      row["strategy"],
            "rule":          row["rule"],
            "rule_name":     row["rule_name"],
            "slot_cluster":  row["slot_cluster"],
            "label":         f"S{int(row['strategy'])} - {RULE_NAMES[int(row['rule'])]}",
            "OV_mean":       row["OV_mean"],
            "OV_hw":         row["OV_hw"],
            "delta_OV":      delta,
            "hw_delta_OV":   hw_delta,
            "sig_OV":        abs(delta) > hw_delta,
        })

    contrasts = pd.DataFrame(rows)
    contrasts["slot_cluster_ord"] = pd.Categorical(
        contrasts["slot_cluster"], categories=CLUSTER_ORDER, ordered=True)

    fp = os.path.join(PLOT_DIR, "step3_contrasts_vs_base.csv")
    contrasts.to_csv(fp, index=False, float_format="%.6f")
    print(f"  Contrasts saved -> {fp}")

    # Forest plot: one panel per cluster, rows = individual slot x combo cells
    # (not averaged across slots — so you see every slot within the cluster)
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=False)
    fig.suptitle(f"OV delta vs. base (S{BASE_STRATEGY}, {BASE_SLOTS} slots)  "
                 f"[negative = better than base]", fontsize=12)

    for ax, cluster in zip(axes, CLUSTER_ORDER):
        sub = (contrasts[contrasts["slot_cluster"] == cluster]
               .copy()
               .assign(row_label=lambda d:
                       d["urgent_slots"].astype(str) + " | " + d["label"])
               .sort_values("delta_OV"))
        y = range(len(sub))
        bar_colors = ["#D85A30" if sig else "#B4B2A9" for sig in sub["sig_OV"]]
        ax.axvline(0, color="black", lw=0.9, ls="--", zorder=1)
        ax.barh(list(y), sub["delta_OV"], xerr=sub["hw_delta_OV"],
                color=bar_colors, alpha=0.82,
                error_kw={"elinewidth": 0.7, "capsize": 2}, zorder=2)
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["row_label"], fontsize=6)
        ax.set_title(cluster, fontsize=11, color=CLUSTER_COLORS[cluster])
        ax.set_xlabel("Delta OV")

    fig.legend(handles=[Patch(color="#D85A30", alpha=0.85, label="Significant"),
                        Patch(color="#B4B2A9", alpha=0.85, label="Not significant")],
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    plt.tight_layout()
    fp = os.path.join(PLOT_DIR, "step3_OV_forest_plot.png")
    plt.savefig(fp, bbox_inches="tight"); plt.close()
    print(f"  Saved {fp}")

    # Summary
    print("\n  -- Combinations that significantly IMPROVE OV vs. base --")
    better = (contrasts[contrasts["sig_OV"] & (contrasts["delta_OV"] < 0)]
              [["slot_cluster", "urgent_slots", "label", "delta_OV", "hw_delta_OV"]]
              .sort_values(["slot_cluster_ord", "delta_OV"]))
    print(better.to_string(index=False) if not better.empty
          else "  None.")

    print("\n  -- Combinations that are significantly WORSE on OV vs. base --")
    worse = (contrasts[contrasts["sig_OV"] & (contrasts["delta_OV"] > 0)]
             [["slot_cluster", "urgent_slots", "label", "delta_OV", "hw_delta_OV"]]
             .sort_values(["slot_cluster_ord", "delta_OV"], ascending=[True, False]))
    print(worse.to_string(index=False) if not worse.empty
          else "  None.")

    return contrasts

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = load_all()
    if df.empty:
        print("No data loaded. Check OUTPUT_DIR and file naming.")
        sys.exit(1)

    print("\n" + "="*60)
    print("STEP 1 -- OV trend plots")
    print("="*60)
    plot_1a(df)
    plot_1b(df)
    plot_1c(df)

    print("\n" + "="*60)
    print("STEP 2 -- OV rankings per cluster")
    print("="*60)
    rankings = rank_combinations(df)

    print("\n" + "="*60)
    print(f"STEP 3 -- OV contrast vs. base (S{BASE_STRATEGY}, {BASE_SLOTS} slots)")
    print("="*60)
    contrasts = contrast_vs_base(df)

    out_csv = os.path.join(PLOT_DIR, "phase1_aggregated_results.csv")
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nFull table -> {out_csv}")
    print(f"All outputs -> {PLOT_DIR}/")