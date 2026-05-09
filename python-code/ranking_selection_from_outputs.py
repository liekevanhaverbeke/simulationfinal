from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# Instellingen
# ============================================================

OUTPUT_DIR = Path("output")
SAVE_DIR = Path("python-code/ranking_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 100

# Kandidaten uit screening
CANDIDATES = [
    ("S2-12-rule2", 2, 12, 2),
    ("S2-14-rule2", 2, 14, 2),
    ("S2-12-rule3", 2, 12, 3),
    ("S2-14-rule3", 2, 14, 3),

    # optioneel: base design toevoegen
    ("S1-14-rule1", 1, 14, 1),
]

# Ranking-and-selection parameters zoals in het voorbeeldrapport
N0 = 20
D_STAR = 0.02
H = 2.747


# ============================================================
# Helpers
# ============================================================

def get_output_path(strategy, slots, rule):
    return OUTPUT_DIR / f"output-S{strategy}-{slots}-rule{rule}.csv"


def read_weekly_ov(strategy, slots, rule, warmup=100):
    path = get_output_path(strategy, slots, rule)

    if not path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {path}")

    df = pd.read_csv(path)

    required = {"week", "elAppWT", "urScanWT", "OT"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} mist kolommen: {missing}")

    df = df[df["week"] >= warmup].copy()

    if df.empty:
        raise ValueError(f"Geen data over na warmup={warmup} voor {path}")

    df["OV"] = df["elAppWT"] / 168 + df["urScanWT"] / 9

    return df[["week", "elAppWT", "urScanWT", "OT", "OV"]].reset_index(drop=True)


def make_batch_means(values, n_batches):
    """
    Splitst een reeks observaties in n_batches gelijke batches
    en neemt per batch het gemiddelde.

    Eventuele leftover observaties aan het einde worden genegeerd.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n_batches > n:
        raise ValueError(
            f"Kan geen {n_batches} batches maken uit slechts {n} observaties."
        )

    batch_size = n // n_batches
    usable_n = batch_size * n_batches

    trimmed = values[:usable_n]
    batches = trimmed.reshape(n_batches, batch_size).mean(axis=1)

    return batches


def evaluate_candidate(name, strategy, slots, rule):
    """
    Voert de two-stage ranking-berekeningen uit voor één design.
    """

    df = read_weekly_ov(strategy, slots, rule, warmup=WARMUP)

    # -------------------------------
    # Stage 1: n0 = 20 batch means
    # -------------------------------
    stage1_batches = make_batch_means(df["OV"].values, N0)

    xbar_n0 = stage1_batches.mean()
    s2_n0 = stage1_batches.var(ddof=1)

    # Formule uit ranking-and-selection:
    # N_i = max{n0 + 1, ceil(h^2 * S_i^2 / (n0 * d*^2))}
    n_required = max(
        N0 + 1,
        int(np.ceil((H ** 2 * s2_n0) / (N0 * D_STAR ** 2)))
    )

    # -------------------------------
    # Stage 2: gebruik N_i batch means
    # -------------------------------
    # Als je genoeg weken hebt, maken we N_i batches uit dezelfde output.
    # Als N_i groter is dan wat praktisch kan, krijg je een duidelijke fout.
    stage2_batches = make_batch_means(df["OV"].values, n_required)

    xbar_ni = stage2_batches.mean()
    s2_ni = stage2_batches.var(ddof=1)

    # Extra metrics over volledige post-warmup data
    mean_el = df["elAppWT"].mean()
    mean_ur = df["urScanWT"].mean()
    mean_ot = df["OT"].mean()
    mean_ov_weekly = df["OV"].mean()

    return {
        "design": name,
        "strategy": strategy,
        "slots": slots,
        "rule": rule,

        "n_weeks_post_warmup": len(df),

        "n0": N0,
        "xbar_n0": xbar_n0,
        "s2_n0": s2_n0,

        "N_required": n_required,
        "xbar_N": xbar_ni,
        "s2_N": s2_ni,

        "mean_elAppWT": mean_el,
        "mean_urScanWT": mean_ur,
        "mean_OT": mean_ot,
        "mean_OV_weekly": mean_ov_weekly,
    }


# ============================================================
# Main
# ============================================================

def main():
    rows = []

    for name, strategy, slots, rule in CANDIDATES:
        print(f"Ranking candidate: {name}")
        row = evaluate_candidate(name, strategy, slots, rule)
        rows.append(row)

    ranking_df = pd.DataFrame(rows)

    # Ranking volgens two-stage estimate
    ranking_df = ranking_df.sort_values("xbar_N").reset_index(drop=True)
    ranking_df["ranking"] = np.arange(1, len(ranking_df) + 1)

    cols = [
        "ranking",
        "design",
        "strategy",
        "slots",
        "rule",
        "xbar_n0",
        "s2_n0",
        "N_required",
        "xbar_N",
        "s2_N",
        "mean_elAppWT",
        "mean_urScanWT",
        "mean_OT",
        "mean_OV_weekly",
        "n_weeks_post_warmup",
    ]

    ranking_df = ranking_df[cols]

    print("\n" + "=" * 80)
    print("FINAL RANKING")
    print("=" * 80)
    print(ranking_df.to_string(index=False))

    output_path = SAVE_DIR / "ranking_selection_results.csv"
    ranking_df.to_csv(output_path, index=False)

    print("\nResultaten opgeslagen in:")
    print(output_path)


if __name__ == "__main__":
    main()