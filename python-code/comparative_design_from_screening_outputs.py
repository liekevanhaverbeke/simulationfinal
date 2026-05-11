"""
All Pairwise Comparison Analysis
==================================
Step 1: Compare all screening candidates vs base (Bonferroni, c = k-1).
Step 2: All pairwise comparison among candidates significantly different
        from base (Bonferroni, c = k*(k-1)/2).

Overall confidence level: 95%  (alpha = 0.05)
Output metric: OV = elAppWT / 168 + urScanWT / 9

Outputs saved to: python-code/comparative_output/
  - step1_vs_base.csv
  - step1_vs_base.png
  - step2_pairwise.csv
  - step2_pairwise.png
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# ─────────────────────────────────────────────
# CONFIG – adjust these paths if needed
# ─────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, '..', 'output')       # simulation output CSVs
COMP_OUT_DIR = os.path.join(SCRIPT_DIR, 'comparative_output') # results go here

SCREENING_CSV = os.path.join(
    SCRIPT_DIR,
    'screening_output',
    'screening_selected_candidate_designs.csv'
)

BASE_TAG   = 'S1-14'
BASE_RULE  = 1
ALPHA_OVERALL = 0.05


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_tag(tag: str) -> tuple[int, int]:
    """
    Parse a tag like 'S2-14' into:
      strategy = 2
      slots    = 14
    """
    try:
        strategy_part, slots_part = tag.split('-')
        strategy = int(strategy_part.replace('S', ''))
        slots = int(slots_part)
        return strategy, slots
    except Exception:
        raise ValueError(f"Cannot parse tag '{tag}'. Expected format like 'S2-14'.")


def make_label(tag: str, rule: int) -> str:
    """
    Label used in plots.
    Example: S=2, N=14, R=1
    """
    strategy, slots = parse_tag(tag)
    return f"S={strategy}, N={slots}, R={rule}"


def load_candidates(screening_csv: str) -> list[tuple[str, int]]:
    if not os.path.exists(screening_csv):
        raise FileNotFoundError(
            f"Screening CSV not found: {screening_csv}\n"
            "Update SCREENING_CSV in the CONFIG section."
        )

    df = pd.read_csv(screening_csv)
    candidates = []

    if 'design' in df.columns:
        for design in df['design']:
            if '-rule' in design:
                tag, rule_str = design.rsplit('-rule', 1)
                candidates.append((tag, int(rule_str)))
            else:
                raise ValueError(
                    f"Cannot parse design '{design}'. Expected format like 'S2-12-rule2'."
                )

    elif {'strategy', 'slots', 'rule'}.issubset(df.columns):
        for _, row in df.iterrows():
            tag = f"S{int(row['strategy'])}-{int(row['slots'])}"
            rule = int(row['rule'])
            candidates.append((tag, rule))

    else:
        raise KeyError(
            "Expected either a 'design' column or 'strategy'/'slots'/'rule' columns. "
            f"Got: {list(df.columns)}"
        )

    if not candidates:
        raise ValueError(f"No candidates found in {screening_csv}.")

    return candidates


def load_weekly_ov(output_dir: str, tag: str, rule: int) -> np.ndarray:
    pattern = os.path.join(output_dir, f'output-{tag}-rule{rule}.csv')
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No file matching: {pattern}")

    df = pd.read_csv(matches[0])

    if 'OV' in df.columns:
        return df['OV'].values

    if 'elAppWT' in df.columns and 'urScanWT' in df.columns:
        return df['elAppWT'].values / 168 + df['urScanWT'].values / 9

    raise KeyError(f"Cannot compute OV from columns: {list(df.columns)}")


def paired_diff_ci(arr_i: np.ndarray, arr_j: np.ndarray, alpha_per: float) -> dict:
    """
    Computes CI for paired difference arr_i - arr_j.
    Lower OV is better.
    """
    diff = arr_i - arr_j
    n = len(diff)

    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)

    t_crit = stats.t.ppf(1 - alpha_per / 2, df=n - 1)
    hw = t_crit * se

    return {
        'mean_diff': mean,
        'hw': hw,
        'lower': mean - hw,
        'upper': mean + hw,
        'n': n
    }


# ─────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────

def plot_ci_chart(
    chart_rows: list[dict],
    title: str,
    xlabel: str,
    filepath: str,
    conf_pct: float,
    is_pairwise: bool = False
):
    """
    Forest-plot style CI chart.

    chart_rows: list of dicts with keys:
      label, mean_diff, lower, upper, significant

    Colour interpretation:
      Green: CI fully below 0
      Red:   CI fully above 0
      Blue:  CI contains 0

    For Step 1:
      Difference = candidate - base.
      Green means candidate is better than base.

    For Step 2:
      Difference = i - j.
      Green means i is better than j.
      Red means j is better than i.
    """
    n = len(chart_rows)

    fig, ax = plt.subplots(figsize=(12, 1.45 * n + 2.4))
    fig.patch.set_facecolor('#F7F7F7')
    ax.set_facecolor('#F7F7F7')

    color_better = '#27AE60'   # green
    color_worse  = '#C0392B'   # red
    color_nosig  = '#2980B9'   # blue

    y_positions = list(range(n - 1, -1, -1))

    all_vals = [r['lower'] for r in chart_rows] + [r['upper'] for r in chart_rows]
    x_min = min(all_vals)
    x_max = max(all_vals)
    x_range = x_max - x_min if x_max != x_min else 1

    for row, ypos in zip(chart_rows, y_positions):
        mean = row['mean_diff']
        lo = row['lower']
        hi = row['upper']

        if hi < 0:
            color = color_better
        elif lo > 0:
            color = color_worse
        else:
            color = color_nosig

        # CI line and caps
        ax.hlines(ypos, lo, hi, colors=color, linewidth=2.8, zorder=3)
        ax.vlines([lo, hi], ypos - 0.22, ypos + 0.22, colors=color, linewidth=2.2, zorder=3)

        # Mean dot
        ax.scatter(mean, ypos, color=color, zorder=5, s=70)

        # CI bound labels
        ax.text(
            lo - x_range * 0.01,
            ypos + 0.32,
            f"{lo:+.4f}",
            ha='right',
            va='bottom',
            fontsize=7.5,
            color=color,
            alpha=0.85
        )

        ax.text(
            hi + x_range * 0.01,
            ypos + 0.32,
            f"{hi:+.4f}",
            ha='left',
            va='bottom',
            fontsize=7.5,
            color=color,
            alpha=0.85
        )

    ax.axvline(0, color='#333333', linewidth=1.3, linestyle='--', alpha=0.75)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['label'] for r in chart_rows], fontsize=9.5)

    ax.tick_params(axis='x', labelsize=8.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=14)

    # Compact x-axis because there is no verdict text on the right anymore
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)

    if is_pairwise:
        green_patch = mpatches.Patch(color=color_better, label='Xi significantly better')
        red_patch = mpatches.Patch(color=color_worse, label='Xj significantly better')
        blue_patch = mpatches.Patch(color=color_nosig, label='No significant difference')
    else:
        green_patch = mpatches.Patch(color=color_better, label='Candidate significantly better than base')
        red_patch = mpatches.Patch(color=color_worse, label='Candidate significantly worse than base')
        blue_patch = mpatches.Patch(color=color_nosig, label='No significant difference')

    ax.legend(
        handles=[green_patch, red_patch, blue_patch],
        fontsize=8.5,
        loc='lower left',
        framealpha=0.85
    )

    fig.text(
        0.5,
        0.01,
        f"Overall {int((1 - ALPHA_OVERALL) * 100)}% CI  |  Bonferroni corrected  |  "
        f"Per-comparison level: {conf_pct:.2f}%",
        ha='center',
        fontsize=8,
        color='#666666'
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Chart saved : {filepath}")


# ─────────────────────────────────────────────
# STEP 1 – comparison vs base
# ─────────────────────────────────────────────

def compare_vs_base(candidates: list[tuple[str, int]], base_ov: np.ndarray) -> list:
    k = len(candidates) + 1   # total systems incl. base
    c = k - 1
    alpha_per = ALPHA_OVERALL / c
    conf_pct = (1 - alpha_per) * 100

    print(f"\n{'=' * 65}")
    print("STEP 1 – Comparison vs Base")
    print(f"  Base         : {BASE_TAG}, rule {BASE_RULE}")
    print(f"  k (total)    : {k}  →  c = k-1 = {c} comparisons")
    print(f"  Per-CI α     : {ALPHA_OVERALL}/{c} = {alpha_per:.6f}")
    print(f"  Per-CI level : {conf_pct:.4f}%")
    print(f"{'=' * 65}")
    print(
        f"{'i':<22}  {'Xi – Xs':>9}  {'Half length':>11}  "
        f"{'CI-':>8}  {'CI+':>8}  Significant?"
    )
    print("-" * 85)

    rows = []
    chart_rows = []
    significant = []

    base_strategy, base_slots = parse_tag(BASE_TAG)
    base_label = make_label(BASE_TAG, BASE_RULE)

    for tag, rule in candidates:
        cand_ov = load_weekly_ov(OUTPUT_DIR, tag, rule)

        n = min(len(base_ov), len(cand_ov))
        res = paired_diff_ci(cand_ov[:n], base_ov[:n], alpha_per)

        strategy, slots = parse_tag(tag)
        label = make_label(tag, rule)

        is_sig = (res['upper'] < 0) or (res['lower'] > 0)
        sig_str = "YES" if is_sig else "no"

        print(
            f"{label:<22}  {res['mean_diff']:>9.4f}  {res['hw']:>11.4f}  "
            f"{res['lower']:>8.4f}  {res['upper']:>8.4f}  {sig_str}"
        )

        rows.append({
            'candidate_strategy': strategy,
            'candidate_slots': slots,
            'candidate_rule': rule,
            'candidate_label': label,

            'base_strategy': base_strategy,
            'base_slots': base_slots,
            'base_rule': BASE_RULE,
            'base_label': base_label,

            'Xi - Xs': round(res['mean_diff'], 4),
            'Half length': round(res['hw'], 4),
            'CI-': round(res['lower'], 4),
            'CI+': round(res['upper'], 4),
            'Significant': sig_str,
        })

        chart_rows.append({
            'label': label,
            'mean_diff': res['mean_diff'],
            'lower': res['lower'],
            'upper': res['upper'],
            'significant': sig_str,
        })

        if is_sig:
            significant.append({
                'tag': tag,
                'strategy': strategy,
                'slots': slots,
                'rule': rule,
                'label': label,
                'ov': cand_ov
            })

    print("-" * 85)
    print(f"\n→ {len(significant)} candidate(s) significantly different from base.")

    # CSV
    csv_path = os.path.join(COMP_OUT_DIR, 'step1_vs_base.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV saved   : {csv_path}")

    # Chart
    plot_ci_chart(
        chart_rows=chart_rows,
        title=f"Step 1 – Candidates vs Base ({base_label})\nDifference in OV  (Xi − Xs)",
        xlabel="Difference in OV  (Xi − Xs)",
        filepath=os.path.join(COMP_OUT_DIR, 'step1_vs_base.png'),
        conf_pct=conf_pct,
        is_pairwise=False,
    )

    return significant


# ─────────────────────────────────────────────
# STEP 2 – all pairwise
# ─────────────────────────────────────────────

def pairwise_comparison(significant_candidates: list):
    k = len(significant_candidates)

    if k < 2:
        print("\nFewer than 2 significant candidates – no pairwise comparison needed.")
        return

    c = k * (k - 1) // 2
    alpha_per = ALPHA_OVERALL / c
    conf_pct = (1 - alpha_per) * 100

    print(f"\n{'=' * 65}")
    print("STEP 2 – All Pairwise Comparison")
    print(f"  Significant candidates : {k}")
    print(f"  Pairs c = k*(k-1)/2    : {c}")
    print(f"  Per-CI α               : {ALPHA_OVERALL}/{c} = {alpha_per:.6f}")
    print(f"  Per-CI level           : {conf_pct:.4f}%")
    print(f"{'=' * 65}")
    print(
        f"{'i':<22}  {'j':<22}  {'Xi – Xj':>9}  {'Half length':>11}  "
        f"{'CI-':>8}  {'CI+':>8}  Significant?"
    )
    print("-" * 105)

    rows = []
    chart_rows = []

    for sys_i, sys_j in itertools.combinations(significant_candidates, 2):
        n = min(len(sys_i['ov']), len(sys_j['ov']))
        res = paired_diff_ci(sys_i['ov'][:n], sys_j['ov'][:n], alpha_per)

        li = sys_i['label']
        lj = sys_j['label']

        is_sig = (res['upper'] < 0) or (res['lower'] > 0)
        sig_str = "YES" if is_sig else "no"

        print(
            f"{li:<22}  {lj:<22}  {res['mean_diff']:>9.4f}  "
            f"{res['hw']:>11.4f}  {res['lower']:>8.4f}  "
            f"{res['upper']:>8.4f}  {sig_str}"
        )

        rows.append({
            'i_strategy': sys_i['strategy'],
            'i_slots': sys_i['slots'],
            'i_rule': sys_i['rule'],
            'i_label': li,

            'j_strategy': sys_j['strategy'],
            'j_slots': sys_j['slots'],
            'j_rule': sys_j['rule'],
            'j_label': lj,

            'Xi - Xj': round(res['mean_diff'], 4),
            'Half length': round(res['hw'], 4),
            'CI-': round(res['lower'], 4),
            'CI+': round(res['upper'], 4),
            'Significant': sig_str,
        })

        chart_rows.append({
            'label': f"{li} vs {lj}",
            'mean_diff': res['mean_diff'],
            'lower': res['lower'],
            'upper': res['upper'],
            'significant': sig_str,
        })

    print("-" * 105)

    print("\nInterpretation (lower OV is better):")
    for r in rows:
        lo = r['CI-']
        hi = r['CI+']
        i = r['i_label']
        j = r['j_label']

        if hi < 0:
            print(f"  ✓ {i} is BETTER than {j}")
        elif lo > 0:
            print(f"  ✓ {j} is BETTER than {i}")
        else:
            print(f"  ~ {i} and {j} are NOT significantly different")

    # CSV
    csv_path = os.path.join(COMP_OUT_DIR, 'step2_pairwise.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  CSV saved   : {csv_path}")

    # Chart
    plot_ci_chart(
        chart_rows=chart_rows,
        title="Step 2 – All Pairwise Comparison\nDifference in OV  (Xi − Xj)",
        xlabel="Difference in OV  (Xi − Xj)",
        filepath=os.path.join(COMP_OUT_DIR, 'step2_pairwise.png'),
        conf_pct=conf_pct,
        is_pairwise=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(COMP_OUT_DIR, exist_ok=True)

    candidates = load_candidates(SCREENING_CSV)

    print(f"\nLoaded {len(candidates)} candidate(s) from screening CSV:")
    for tag, rule in candidates:
        print(f"  {make_label(tag, rule)}")

    base_ov = load_weekly_ov(OUTPUT_DIR, BASE_TAG, BASE_RULE)

    print(
        f"\nBase ({make_label(BASE_TAG, BASE_RULE)}): "
        f"{len(base_ov)} weeks, mean OV = {base_ov.mean():.4f}"
    )

    significant = compare_vs_base(candidates, base_ov)
    pairwise_comparison(significant)

    print(f"\nAll outputs saved to: {COMP_OUT_DIR}")
    print("Done.")


if __name__ == '__main__':
    main()