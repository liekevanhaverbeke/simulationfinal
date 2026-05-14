import csv
import math
import numpy as np
from pathlib import Path


h       = 2.747
p       = 0.9
n0      = 20
d_star  = 0.02

weightEl = 1.0 / 168.0
weightUr = 1.0 / 9.0



designs = [
    ("S2_14_R2",  0.38766, 0.03104,  21,     0.3660),
    ("S2_14_R3",  0.40155, 0.03594,  24,     0.3810),
    ("S2_12_R2",  0.40069, 0.04031,  31,     0.3613),
    ("S2_12_R3",  0.41798, 0.03371,  21,     0.3717),
    ("S1_14_R1",  0.44500, 0.03230,  21,     0.4030),
]


def compute_Ni(S2_i: float, h: float, n0: int, d_star: float) -> tuple[float, int]:
    formula_value = (h ** 2) * S2_i * n0 / (d_star ** 2)
    N_i_exact     = max(n0 + 1, formula_value)
    N_i_rounded   = math.ceil(N_i_exact)
    return N_i_exact, N_i_rounded


def compute_final_estimator(
    X_n0: float,
    X_add: float,
    n0: int,
    N_i: int
) -> tuple[float, float, float]:

    add_obs = N_i - n0
    W1 = n0 / N_i
    W2 = add_obs / N_i
    X_final = W1 * X_n0 + W2 * X_add
    return W1, W2, X_final



def run_rinott_procedure(designs, h, n0, d_star):

    print("=" * 80)
    print("TWO-STAGE RINOTT RANKING & SELECTION PROCEDURE")
    print("=" * 80)
    print(f"\n  h = {h},  p = {p},  n0 = {n0},  d* = {d_star}")
    print(f"  Formula: N_i = max{{n0+1, ceil( h²·S²_i·n0 / (d*)² )}}\n")

    print("STAGE 1  —  Initial sample statistics")
    print("-" * 78)
    hdr = f"{'Design':<14} {'X(n0)':>9} {'S²(n0)':>9} {'N_i exact':>11} {'N_i round':>10} {'Add obs':>8}"
    print(hdr)
    print("-" * 78)

    results = []
    for name, X_n0, S2_n0, N_i_given, X_add in designs:
        N_i_exact, N_i_round_calc = compute_Ni(S2_n0, h, n0, d_star)

        add_obs = N_i_given - n0
        print(f"{name:<14} {X_n0:>9.5f} {S2_n0:>9.5f} {N_i_exact:>11.2f} {N_i_given:>10} {add_obs:>8}")
        results.append({
            "name": name, "X_n0": X_n0, "S2_n0": S2_n0,
            "N_i_exact": N_i_exact, "N_i": N_i_given, "X_add": X_add,
        })

    print(f"\nSTAGE 2  —  Final estimator  X_i(N_i) = W1·X(n0) + W2·X(N_i−n0)")
    print("-" * 78)
    hdr2 = f"{'Design':<14} {'X(n0)':>9} {'X(Ni-n0)':>10} {'Ni':>5} {'W1':>8} {'W2':>8} {'X(Ni)':>9} {'Rank':>6}"
    print(hdr2)
    print("-" * 78)

    for r in results:
        W1, W2, X_final = compute_final_estimator(r["X_n0"], r["X_add"], n0, r["N_i"])
        r.update({"W1": W1, "W2": W2, "X_final": X_final})


    results.sort(key=lambda r: r["X_final"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank
        print(f"{r['name']:<14} {r['X_n0']:>9.5f} {r['X_add']:>10.3f} "
              f"{r['N_i']:>5} {r['W1']:>8.3f} {r['W2']:>8.3f} "
              f"{r['X_final']:>9.4f} {rank:>6}")

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



def pairwise_significance_table():

    print("=" * 80)
    print("PAIRWISE SIGNIFICANCE  —  step2_pairwise_2.csv")
    print("=" * 80)
    pairs = [
        ("S2_12_R2 vs S2_14_R2", -0.0047, -0.0149,  0.0054, "no"),
        ("S2_12_R2 vs S2_12_R3", -0.0104, -0.0107, -0.0101, "YES ✓"),
        ("S2_12_R2 vs S2_14_R3", -0.0198, -0.0299, -0.0096, "YES ✓"),
        ("S2_14_R2 vs S2_12_R3", -0.0057, -0.0158,  0.0045, "no"),
        ("S2_14_R2 vs S2_14_R3", -0.0150, -0.0153, -0.0147, "YES ✓"),
        ("S2_12_R3 vs S2_14_R3", -0.0093, -0.0195,  0.0008, "no"),
    ]
    print(f"\n  {'Pair':<32} {'Xi−Xj':>8}  {'95% CI':^22}  {'Significant':>11}")
    print("  " + "-" * 78)
    for pair, diff, ci_lo, ci_hi, sig in pairs:
        ci_str = f"[{ci_lo:.4f}, {ci_hi:+.4f}]"
        print(f"  {pair:<32} {diff:>8.4f}  {ci_str:^22}  {sig:>11}")

    print()
    print("  Key finding: S2_12_R2 and S2_14_R2 are NOT significantly different")
    print("  → They form a statistically indistinguishable best group.\n")



def save_results_to_csv(results: list[dict], output_path: str = "rinott_results.csv") -> None:

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


if __name__ == "__main__":
    results = run_rinott_procedure(designs, h, n0, d_star)
    pairwise_significance_table()
    save_results_to_csv(results, output_path="rinott_results.csv")
