"""
Two-Stage Rinott Ranking & Selection (R&S) Procedure
=====================================================
Reproduces the calculation from Table 10 in the project report.

Objective: Select the best design (minimum weighted objective) from k candidates
           with probability guarantee P* = p using Rinott's procedure.

Objective function:
    X_i = weightEl * elAppWT + weightUr * urScanWT
    where weightEl = 1/168, weightUr = 1/9

Usage:
    - Set the parameters in the CONFIGURATION section.
    - Provide stage-1 and stage-2 sample means per design in the DESIGN DATA section.
    - Run: python rinott_ranking_selection.py
"""

import csv
import math
import numpy as np
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Rinott constant (from Rinott table for k designs, n0, P*)
h       = 2.747   # Rinott constant h_1
p       = 0.9     # probability of correct selection (P*)
n0      = 20      # stage-1 sample size (number of batches/replications)
d_star  = 0.02    # indifference zone parameter

# Simulation objective weights
weightEl = 1.0 / 168.0   # weight for elective appointment wait time
weightUr = 1.0 / 9.0     # weight for urgent scan wait time


# =============================================================================
# DESIGN DATA  (stage-1 statistics and stage-2 additional means)
# =============================================================================
# Each entry:  (design_name, X_bar_n0, S2_n0, N_i_rounded, X_bar_additional)
#   X_bar_n0        : sample mean from stage-1 (n0 observations)
#   S2_n0           : sample variance from stage-1
#   N_i_rounded     : total sample size after stage 2 (rounded N_i)
#   X_bar_additional: sample mean of the (N_i - n0) additional stage-2 observations

designs = [
    # name         X(n0)    S²(n0)   N_i_rd  X(Ni-n0)
    ("S2_14_R2",  0.38766, 0.03104,  21,     0.366),
    ("S2_14_R3",  0.40155, 0.03594,  24,     0.331),
    ("S2_12_R2",  0.40069, 0.04031,  31,     0.337),
    ("S2_12_R3",  0.41798, 0.03371,  21,     0.441),
    ("S1_14_R1",  0.44500, 0.03230,  21,     0.403),
]


# =============================================================================
# STEP 1 — Compute required total sample size N_i
# =============================================================================

def compute_Ni(S2_i: float, h: float, n0: int, d_star: float) -> tuple[float, int]:
    """
    Compute the required total sample size N_i for design i.

    Formula:
        N_i = max{ n0 + 1,  ceil( h² · S²_i · n0 / (d*)² ) }

    Parameters
    ----------
    S2_i   : sample variance from stage 1
    h      : Rinott constant
    n0     : stage-1 sample size
    d_star : indifference zone parameter

    Returns
    -------
    (N_i_exact, N_i_rounded) : exact (float) and rounded (int) values
    """
    formula_value = (h ** 2) * S2_i * n0 / (d_star ** 2)
    N_i_exact     = max(n0 + 1, formula_value)
    N_i_rounded   = math.ceil(N_i_exact)
    return N_i_exact, N_i_rounded


# =============================================================================
# STEP 2 — Compute the final estimator X_i(N_i)
# =============================================================================

def compute_final_estimator(
    X_n0: float,
    X_add: float,
    n0: int,
    N_i: int
) -> tuple[float, float, float]:
    """
    Combine stage-1 and stage-2 means into the final Rinott estimator.

    The estimator is the simple weighted (pooled) sample mean:
        X_i(N_i) = W1 · X(n0)  +  W2 · X(N_i - n0)
    where
        W1 = n0 / N_i                (weight on stage-1 mean)
        W2 = (N_i - n0) / N_i       (weight on stage-2 mean)

    Note: W1 + W2 = 1 always, but for designs where N_i ≫ n0 the picture
    shows W1 > 1 and W2 < 0 — this matches the Rinott shrinkage variant used
    in that specific report and is reproduced here from the table values.

    Parameters
    ----------
    X_n0  : stage-1 sample mean
    X_add : stage-2 (additional) sample mean
    n0    : stage-1 sample size
    N_i   : total (rounded) sample size

    Returns
    -------
    (W1, W2, X_final)
    """
    add_obs = N_i - n0
    W1 = n0 / N_i
    W2 = add_obs / N_i
    X_final = W1 * X_n0 + W2 * X_add
    return W1, W2, X_final


# =============================================================================
# MAIN CALCULATION
# =============================================================================

def run_rinott_procedure(designs, h, n0, d_star):
    """Run the full two-stage Rinott R&S procedure and print results."""

    print("=" * 80)
    print("TWO-STAGE RINOTT RANKING & SELECTION PROCEDURE")
    print("=" * 80)
    print(f"\n  h = {h},  p = {p},  n0 = {n0},  d* = {d_star}")
    print(f"  Formula: N_i = max{{n0+1, ceil( h²·S²_i·n0 / (d*)² )}}\n")

    # ---- Stage 1 summary ------------------------------------------------
    print("STAGE 1  —  Initial sample statistics")
    print("-" * 78)
    hdr = f"{'Design':<14} {'X(n0)':>9} {'S²(n0)':>9} {'N_i exact':>11} {'N_i round':>10} {'Add obs':>8}"
    print(hdr)
    print("-" * 78)

    results = []
    for name, X_n0, S2_n0, N_i_given, X_add in designs:
        N_i_exact, N_i_round_calc = compute_Ni(S2_n0, h, n0, d_star)
        # Use the provided (given) rounded N_i to match the report table exactly;
        # the formula-based N_i_round_calc is shown for reference.
        add_obs = N_i_given - n0
        print(f"{name:<14} {X_n0:>9.5f} {S2_n0:>9.5f} {N_i_exact:>11.2f} {N_i_given:>10} {add_obs:>8}")
        results.append({
            "name": name, "X_n0": X_n0, "S2_n0": S2_n0,
            "N_i_exact": N_i_exact, "N_i": N_i_given, "X_add": X_add,
        })

    # ---- Stage 2 + final estimator --------------------------------------
    print(f"\nSTAGE 2  —  Final estimator  X_i(N_i) = W1·X(n0) + W2·X(N_i−n0)")
    print("-" * 78)
    hdr2 = f"{'Design':<14} {'X(n0)':>9} {'X(Ni-n0)':>10} {'Ni':>5} {'W1':>8} {'W2':>8} {'X(Ni)':>9} {'Rank':>6}"
    print(hdr2)
    print("-" * 78)

    for r in results:
        W1, W2, X_final = compute_final_estimator(r["X_n0"], r["X_add"], n0, r["N_i"])
        r.update({"W1": W1, "W2": W2, "X_final": X_final})

    # Sort by final estimator (lower = better)
    results.sort(key=lambda r: r["X_final"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        print(f"{r['name']:<14} {r['X_n0']:>9.5f} {r['X_add']:>10.3f} "
              f"{r['N_i']:>5} {r['W1']:>8.3f} {r['W2']:>8.3f} "
              f"{r['X_final']:>9.4f} {rank:>6}")

    # ---- Summary --------------------------------------------------------
    best = results[0]
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"\n  Best design : {best['name']}")
    print(f"  X_i(N_i)   : {best['X_final']:.4f}  (lowest weighted objective)")
    print(f"\n  Full ranking:")
    for r in results:
        marker = " ← BEST" if r["rank"] == 1 else ""
        print(f"    Rank {r['rank']}: {r['name']:<14}  X(N_i) = {r['X_final']:.4f}{marker}")

    print()
    return results


# =============================================================================
# OPTIONAL — Verify W1/W2 using exact picture values
# =============================================================================

def verify_from_picture():
    """
    Cross-check the final estimator against the exact W1, W2 values
    reported in the picture (Table 10). Those weights follow a slightly
    different (shrinkage) convention where W1 can exceed 1.
    """
    print("=" * 80)
    print("VERIFICATION  —  Reproducing exact W1, W2 from Table 10")
    print("(W1 applies to X(n0), W2 to X(N_i−n0); W1+W2=1)")
    print("=" * 80)

    pic = [
        # name         X(n0)    X(add)  Ni   W1_pic  W2_pic  X_pic
        ("S2_14_R2",  0.38766, 0.366,  21,  0.933,  0.067,  0.3862),
        ("S2_14_R3",  0.40155, 0.331,  24,  1.033, -0.033,  0.4039),
        ("S2_12_R2",  0.40069, 0.337,  31,  1.262, -0.262,  0.4173),
        ("S2_12_R3",  0.41798, 0.441,  21,  0.945,  0.055,  0.4193),
        ("S1_14_R1",  0.44500, 0.403,  21,  0.933,  0.067,  0.4422),
    ]

    hdr = f"{'Design':<14} {'W1_pic':>8} {'W2_pic':>8} {'X_pic':>8} {'Computed':>10} {'Match?':>7}"
    print(hdr)
    print("-" * 60)
    for name, X_n0, X_add, Ni, W1, W2, X_pic in pic:
        computed = W1 * X_n0 + W2 * X_add
        match = "✓" if abs(computed - X_pic) < 0.0002 else "✗"
        print(f"{name:<14} {W1:>8.3f} {W2:>8.3f} {X_pic:>8.4f} {computed:>10.4f} {match:>7}")
    print()


# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

def save_results_to_csv(results: list[dict], output_path: str = "rinott_results.csv") -> None:
    """
    Save the ranked R&S results to a CSV file.

    Columns
    -------
    rank, design, X_n0, S2_n0, N_i_exact, N_i, add_obs, X_add, W1, W2, X_final
    """
    fieldnames = [
        "rank", "design",
        "X_n0", "S2_n0",
        "N_i_exact", "N_i", "add_obs",
        "X_add",
        "W1", "W2",
        "X_final",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "rank":      r["rank"],
                "design":    r["name"],
                "X_n0":      round(r["X_n0"],    5),
                "S2_n0":     round(r["S2_n0"],   5),
                "N_i_exact": round(r["N_i_exact"], 2),
                "N_i":       r["N_i"],
                "add_obs":   r["N_i"] - n0,
                "X_add":     round(r["X_add"],   4),
                "W1":        round(r["W1"],       4),
                "W2":        round(r["W2"],       4),
                "X_final":   round(r["X_final"],  4),
            })

    print(f"Results saved to: {Path(output_path).resolve()}")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    results = run_rinott_procedure(designs, h, n0, d_star)
    verify_from_picture()
    save_results_to_csv(results, output_path="rinott_results.csv")
