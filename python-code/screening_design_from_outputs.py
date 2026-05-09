"""
screening_design_from_outputs.py

Screening design zoals in het voorbeeldrapport:
- leest bestaande output CSV's, bv. output/output-S1-14-rule1.csv
- berekent OV = elAppWT/168 + urScanWT/9
- voert 2^3 screening designs uit met factoren:
    S = strategy
    R = rule
    N = aantal urgent slots
- fit per screening een full factorial coded linear model:
    OV = beta0 + betaS*S + betaR*R + betaN*N
         + betaSR*S*R + betaSN*S*N + betaRN*R*N + betaSRN*S*R*N
- maakt confidence intervals voor effecten
- markeert effecten significant als het CI 0 niet bevat
- schrijft resultaten weg naar een aparte outputmap

Run vanuit project root:
    python python-code/screening_design_from_outputs.py

Of expliciet:
    python python-code/screening_design_from_outputs.py --output-dir output --warmup 100
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # scipy fallback
    stats = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True, order=True)
class Design:
    strategy: int
    slots: int
    rule: int

    @property
    def name(self) -> str:
        return f"S{self.strategy}-{self.slots}-rule{self.rule}"


def safe_t_critical(confidence: float, df: int) -> float:
    """t critical; fallback naar 1.96 als scipy ontbreekt."""
    alpha = 1.0 - confidence
    if stats is not None and df > 0:
        return float(stats.t.ppf(1 - alpha / 2, df=df))
    return 1.96


# ============================================================
# Input files ontdekken en inlezen
# ============================================================

def discover_output_files(output_dir: Path) -> Dict[Design, Path]:
    """
    Ondersteunde naamstructuur:
        output-S1-14-rule1.csv
        output-S1-14-rule2.csv
        ...
    """
    pattern = re.compile(r"output-S(\d+)-(\d+)-rule(\d+)\.csv$")
    file_map: Dict[Design, Path] = {}

    for path in output_dir.glob("output-S*-*-rule*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        s = int(match.group(1))
        n = int(match.group(2))
        r = int(match.group(3))
        file_map[Design(strategy=s, slots=n, rule=r)] = path

    if not file_map:
        raise FileNotFoundError(
            f"Geen output-S<strategy>-<slots>-rule<rule>.csv bestanden gevonden in {output_dir}"
        )

    return file_map


def read_output_file(path: Path, warmup: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"week", "elAppWT", "urScanWT", "OT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} mist kolommen: {missing}")

    df = df[df["week"] >= warmup].copy()
    if df.empty:
        raise ValueError(f"Geen observaties na warmup={warmup} in {path}")

    df["OV"] = df["elAppWT"] / 168.0 + df["urScanWT"] / 9.0
    return df


def load_design_observations(
    design: Design,
    file_map: Dict[Design, Path],
    warmup: int,
    response: str = "OV",
) -> pd.DataFrame:
    if design not in file_map:
        raise FileNotFoundError(f"Geen outputbestand voor design {design.name}")

    df = read_output_file(file_map[design], warmup=warmup)
    out = pd.DataFrame({
        "design": design.name,
        "strategy": design.strategy,
        "slots": design.slots,
        "rule": design.rule,
        "week": df["week"].values,
        "elAppWT": df["elAppWT"].values,
        "elScanWT": df["elScanWT"].values if "elScanWT" in df.columns else np.nan,
        "urScanWT": df["urScanWT"].values,
        "OT": df["OT"].values,
        "OV": df["OV"].values,
    })

    if response not in out.columns:
        raise ValueError(f"Response '{response}' bestaat niet. Kies uit: {list(out.columns)}")

    return out


# ============================================================
# Screening design definitie
# ============================================================

@dataclass(frozen=True)
class ScreeningSpec:
    name: str
    strategies: Tuple[int, int]
    rules: Tuple[int, int]
    slots: Tuple[int, int]


def default_screening_specs() -> List[ScreeningSpec]:
    """
    Zelfde structuur als in het voorbeeldrapport:
    1: S1/S2, R1/R2, N10/N12
    2: S2/S3, R2/R3, N12/N14
    3: S2/S3, R2/R3, N14/N16
    4: S2/S3, R2/R4, N12/N14
    """
    return [
        ScreeningSpec("screening_1", strategies=(1, 2), rules=(1, 2), slots=(10, 12)),
        ScreeningSpec("screening_2", strategies=(2, 3), rules=(2, 3), slots=(12, 14)),
        ScreeningSpec("screening_3", strategies=(2, 3), rules=(2, 3), slots=(14, 16)),
        ScreeningSpec("screening_4", strategies=(2, 3), rules=(2, 4), slots=(12, 14)),
    ]


def designs_for_spec(spec: ScreeningSpec) -> List[Design]:
    return [
        Design(strategy=s, slots=n, rule=r)
        for s in spec.strategies
        for r in spec.rules
        for n in spec.slots
    ]


def coded_value(value: int, low_high: Tuple[int, int]) -> int:
    low, high = low_high
    if value == low:
        return -1
    if value == high:
        return 1
    raise ValueError(f"Value {value} is niet gelijk aan low/high {low_high}")


def build_screening_dataset(
    spec: ScreeningSpec,
    file_map: Dict[Design, Path],
    warmup: int,
    response: str,
    strict: bool = False,
) -> pd.DataFrame:
    frames = []
    missing = []

    for design in designs_for_spec(spec):
        if design not in file_map:
            missing.append(design.name)
            continue

        df = load_design_observations(design, file_map, warmup=warmup, response=response)
        df["S_code"] = coded_value(design.strategy, spec.strategies)
        df["R_code"] = coded_value(design.rule, spec.rules)
        df["N_code"] = coded_value(design.slots, spec.slots)
        frames.append(df)

    if missing:
        msg = f"{spec.name}: ontbrekende outputfiles: {missing}"
        if strict:
            raise FileNotFoundError(msg)
        print("Waarschuwing:", msg)

    if len(frames) < 8 and strict:
        raise ValueError(f"{spec.name}: onvolledig 2^3 design")

    if not frames:
        raise ValueError(f"{spec.name}: geen enkele designfile gevonden")

    data = pd.concat(frames, ignore_index=True)

    data["S_R"] = data["S_code"] * data["R_code"]
    data["S_N"] = data["S_code"] * data["N_code"]
    data["R_N"] = data["R_code"] * data["N_code"]
    data["S_R_N"] = data["S_code"] * data["R_code"] * data["N_code"]

    return data


# ============================================================
# Full factorial coded regression
# ============================================================

def fit_screening_model(
    data: pd.DataFrame,
    response: str = "OV",
    confidence: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit OLS met gecodeerde effecten.

    Model:
        response = beta0 + betaS*S + betaR*R + betaN*N
                 + betaSR*S*R + betaSN*S*N + betaRN*R*N + betaSRN*S*R*N

    Bij +/- codering is beta de halve verandering van low naar high.
    Het volledige effect high-low is 2*beta. We rapporteren beide.
    """
    terms = ["S_code", "R_code", "N_code", "S_R", "S_N", "R_N", "S_R_N"]
    labels = {
        "S_code": "S: strategy high vs low",
        "R_code": "R: rule high vs low",
        "N_code": "N: urgent slots high vs low",
        "S_R": "S*R interaction",
        "S_N": "S*N interaction",
        "R_N": "R*N interaction",
        "S_R_N": "S*R*N interaction",
    }

    X = data[terms].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    y = data[response].to_numpy(dtype=float)

    beta, residuals, rank, singular_values = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    resid = y - y_hat

    n_obs = len(y)
    p = X_design.shape[1]
    df_resid = max(n_obs - p, 1)
    sse = float(np.sum(resid ** 2))
    mse = sse / df_resid

    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    se_beta = np.sqrt(np.diag(xtx_inv) * mse)
    tcrit = safe_t_critical(confidence, df_resid)

    rows = []
    for idx, term in enumerate(["Intercept"] + terms):
        b = beta[idx]
        se = se_beta[idx]
        lo = b - tcrit * se
        hi = b + tcrit * se

        if term == "Intercept":
            effect = np.nan
            effect_lo = np.nan
            effect_hi = np.nan
            significant = False
            meaning = "Grand mean"
        else:
            effect = 2 * b
            effect_lo = 2 * lo
            effect_hi = 2 * hi
            significant = not (lo <= 0 <= hi)
            meaning = labels[term]

        rows.append({
            "term": term,
            "meaning": meaning,
            "beta": b,
            "se_beta": se,
            "ci_low_beta": lo,
            "ci_high_beta": hi,
            "effect_high_minus_low": effect,
            "ci_low_effect": effect_lo,
            "ci_high_effect": effect_hi,
            "significant": significant,
            "df_resid": df_resid,
            "n_obs": n_obs,
            "mse": mse,
        })

    effects = pd.DataFrame(rows)

    cell_summary = (
        data.groupby(["design", "strategy", "slots", "rule"], as_index=False)
        .agg(
            mean_response=(response, "mean"),
            sd_response=(response, "std"),
            n=(response, "size"),
            mean_elAppWT=("elAppWT", "mean"),
            mean_urScanWT=("urScanWT", "mean"),
            mean_OT=("OT", "mean"),
            mean_OV=("OV", "mean"),
        )
        .sort_values("mean_response")
        .reset_index(drop=True)
    )

    return effects, cell_summary


def interpret_effects(effects: pd.DataFrame, spec: ScreeningSpec) -> List[str]:
    """Korte tekstuele interpretatie per significant main effect."""
    interpretations = []
    mapping = {
        "S_code": ("strategy", spec.strategies[0], spec.strategies[1]),
        "R_code": ("rule", spec.rules[0], spec.rules[1]),
        "N_code": ("urgent slots", spec.slots[0], spec.slots[1]),
    }

    for term, (factor, low, high) in mapping.items():
        row = effects.loc[effects["term"] == term]
        if row.empty:
            continue
        row = row.iloc[0]
        if not bool(row["significant"]):
            interpretations.append(
                f"{factor}: geen significant hoofdeffect tussen {low} en {high}."
            )
            continue

        beta = row["beta"]
        if beta < 0:
            interpretations.append(
                f"{factor}: significant beter om van {low} naar {high} te gaan; OV daalt."
            )
        else:
            interpretations.append(
                f"{factor}: significant slechter om van {low} naar {high} te gaan; OV stijgt."
            )

    return interpretations


# ============================================================
# Plotting
# ============================================================

def plot_effects(effects: pd.DataFrame, title: str, output_path: Path) -> None:
    if plt is None:
        print("Matplotlib niet beschikbaar; plot wordt overgeslagen.")
        return

    eff = effects[effects["term"] != "Intercept"].copy()
    eff = eff.reset_index(drop=True)

    y = np.arange(len(eff))
    x = eff["effect_high_minus_low"].to_numpy(dtype=float)
    xerr_low = x - eff["ci_low_effect"].to_numpy(dtype=float)
    xerr_high = eff["ci_high_effect"].to_numpy(dtype=float) - x
    xerr = np.vstack([xerr_low, xerr_high])

    plt.figure(figsize=(10, 5))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, eff["term"])
    plt.xlabel("Effect on OV: high level - low level")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Hoofdprocedure
# ============================================================

def run_screening_designs(
    output_dir: str = "output",
    results_dir: str | None = None,
    warmup: int = 0,
    response: str = "OV",
    confidence: float = 0.95,
    strict: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir_path = Path(output_dir)
    file_map = discover_output_files(output_dir_path)

    if results_dir is None:
        results_path = Path("python-code") / "screening_output"
    else:
        results_path = Path(results_dir)

    results_path.mkdir(parents=True, exist_ok=True)

    all_effects = []
    all_summaries = []
    all_interpretations = []

    for spec in default_screening_specs():
        print("\n" + "=" * 80)
        print(spec.name)
        print("=" * 80)
        print(f"Strategies: {spec.strategies}, Rules: {spec.rules}, Slots: {spec.slots}")

        data = build_screening_dataset(
            spec=spec,
            file_map=file_map,
            warmup=warmup,
            response=response,
            strict=strict,
        )

        effects, summary = fit_screening_model(
            data=data,
            response=response,
            confidence=confidence,
        )

        effects.insert(0, "screening", spec.name)
        effects.insert(1, "low_strategy", spec.strategies[0])
        effects.insert(2, "high_strategy", spec.strategies[1])
        effects.insert(3, "low_rule", spec.rules[0])
        effects.insert(4, "high_rule", spec.rules[1])
        effects.insert(5, "low_slots", spec.slots[0])
        effects.insert(6, "high_slots", spec.slots[1])

        summary.insert(0, "screening", spec.name)

        interpretations = interpret_effects(effects, spec)
        interp_df = pd.DataFrame({
            "screening": spec.name,
            "interpretation": interpretations,
        })

        all_effects.append(effects)
        all_summaries.append(summary)
        all_interpretations.append(interp_df)

        print("\nCell means:")
        print(summary[["design", "mean_response", "mean_elAppWT", "mean_urScanWT", "mean_OT", "n"]].to_string(index=False))

        print("\nEffects:")
        print(
            effects[effects["term"] != "Intercept"]
            [["term", "effect_high_minus_low", "ci_low_effect", "ci_high_effect", "significant"]]
            .to_string(index=False)
        )

        print("\nInterpretation:")
        for text in interpretations:
            print("-", text)

        spec_prefix = results_path / spec.name
        effects.to_csv(f"{spec_prefix}_effects.csv", index=False)
        summary.to_csv(f"{spec_prefix}_cell_means.csv", index=False)
        data.to_csv(f"{spec_prefix}_raw_observations.csv", index=False)
        plot_effects(effects, f"CI {spec.name}", Path(f"{spec_prefix}_effects_plot.png"))

    effects_df = pd.concat(all_effects, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    interpretations_df = pd.concat(all_interpretations, ignore_index=True)

    effects_df.to_csv(results_path / "all_screening_effects.csv", index=False)
    summary_df.to_csv(results_path / "all_screening_cell_means.csv", index=False)
    interpretations_df.to_csv(results_path / "all_screening_interpretations.csv", index=False)

    # Kandidaten zoals in het voorbeeldrapport: S2, R2/R3, N12/N14 indien aanwezig.
    candidate_designs = [
        Design(strategy=2, slots=12, rule=2),
        Design(strategy=2, slots=12, rule=3),
        Design(strategy=2, slots=14, rule=2),
        Design(strategy=2, slots=14, rule=3),
    ]
    candidate_rows = []
    for d in candidate_designs:
        if d in file_map:
            df = read_output_file(file_map[d], warmup=warmup)
            candidate_rows.append({
                "design": d.name,
                "strategy": d.strategy,
                "slots": d.slots,
                "rule": d.rule,
                "mean_elAppWT": df["elAppWT"].mean(),
                "mean_urScanWT": df["urScanWT"].mean(),
                "mean_OT": df["OT"].mean(),
                "mean_OV": df["OV"].mean(),
                "n_weeks": len(df),
            })

    candidates_df = pd.DataFrame(candidate_rows).sort_values("mean_OV").reset_index(drop=True)
    candidates_df.to_csv(results_path / "screening_selected_candidate_designs.csv", index=False)

    print("\n" + "=" * 80)
    print("SELECTED CANDIDATE DESIGNS")
    print("=" * 80)
    if candidates_df.empty:
        print("Geen van de standaard kandidaatdesigns S2-12/14-rule2/3 werd gevonden.")
    else:
        print(candidates_df.to_string(index=False))

    print("\nResultaten opgeslagen in:")
    print(results_path.resolve())

    return effects_df, summary_df, candidates_df


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--response", type=str, default="OV", choices=["OV", "elAppWT", "urScanWT", "OT", "elScanWT"])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_screening_designs(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        warmup=args.warmup,
        response=args.response,
        confidence=args.confidence,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()