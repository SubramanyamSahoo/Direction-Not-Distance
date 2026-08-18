#!/usr/bin/env python3
"""Analysis for the targeted Norm-Matched follow-up.

The primary follow-up contrast is Coordinate Mortality (existing experiment)
versus Norm-Matched shrinkage (new targeted experiment) on final held-out
preference accuracy.  The two shift regimes form the prespecified primary
family and receive Holm correction across the two p-values.  Preference margin,
plasticity, drift, and behavioral-judge comparisons are secondary/exploratory.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def exact_signflip_paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return {"mean_difference": float("nan"), "p_exact": float("nan"), "n": 0}
    observed = abs(d.mean())
    stats = [abs(np.mean(d * np.asarray(signs))) for signs in itertools.product((-1.0, 1.0), repeat=n)]
    return {
        "mean_difference": float(d.mean()),
        "p_exact": float(np.mean(np.asarray(stats) >= observed - 1e-15)),
        "n": int(n),
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    out = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * p[idx], 1.0)
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "regime", "condition"]
    frame = frame.sort_values(keys + ["phase"]).copy()
    for col in ["alignment_accuracy", "alignment_margin_mean", "shift_nll"]:
        baseline = frame.groupby(keys)[col].transform("first")
        if col == "shift_nll":
            frame["shift_nll_improvement"] = baseline - frame[col]
        else:
            frame[f"{col}_change"] = frame[col] - baseline
    return frame


def exact_bootstrap_ci(values: np.ndarray, confidence: float = 0.95):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # Seven seeds -> 7^7 = 823,543 exact ordered bootstrap means.
    grids = np.indices((n,) * n, dtype=np.int16).reshape(n, -1).T
    means = x[grids].mean(axis=1)
    alpha = 1.0 - confidence
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, default=Path("pci_h100_outputs/results/all_metrics_enriched.csv"))
    p.add_argument("--target", type=Path, default=Path("norm_matched_outputs/results/norm_matched_metrics.csv"))
    p.add_argument("--original-judge", type=Path, default=Path("pci_h100_outputs/results/behavior_judge_aggregate.csv"))
    p.add_argument("--target-judge", type=Path, default=Path("norm_matched_outputs/results/behavior_judge_aggregate.csv"))
    p.add_argument("--out", type=Path, default=Path("norm_matched_outputs/results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(args.original)
    target = enrich(pd.read_csv(args.target))

    expected_seeds = set(range(42, 49))
    expected = {
        (seed, regime, phase)
        for seed in expected_seeds
        for regime in ("benign", "conflict")
        for phase in range(7)
    }
    observed = set(target[["seed", "regime", "phase"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Targeted follow-up incomplete; missing {len(missing)} rows, first={missing[:10]}")

    target.to_csv(args.out / "norm_matched_metrics_enriched.csv", index=False)
    original8 = original[(original.model == "Qwen3-8B") & (original.seed.isin(expected_seeds))].copy()
    comparison = pd.concat(
        [
            original8[original8.condition.isin(["coordinate_mortality", "unconstrained", "global_projection"])],
            target,
        ],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(args.out / "original_plus_norm_matched_all_phases.csv", index=False)
    final = comparison[comparison.phase == 6].copy()
    final.to_csv(args.out / "original_plus_norm_matched_final.csv", index=False)

    # Descriptive exact-bootstrap summaries.
    metrics = [
        "alignment_accuracy_change",
        "alignment_margin_mean_change",
        "shift_nll_improvement",
        "relative_l2_drift",
        "retained_update_norm_ratio",
    ]
    summary_rows = []
    for (regime, condition), group in final.groupby(["regime", "condition"]):
        for metric in metrics:
            mean, lo, hi = exact_bootstrap_ci(group[metric].to_numpy())
            summary_rows.append({
                "regime": regime,
                "condition": condition,
                "metric": metric,
                "mean": mean,
                "ci95_lower_exact_bootstrap": lo,
                "ci95_upper_exact_bootstrap": hi,
                "seeds": int(group.seed.nunique()),
            })
    pd.DataFrame(summary_rows).to_csv(args.out / "norm_matched_group_summaries.csv", index=False)

    # Primary follow-up family: CM vs NM final preference accuracy in benign and conflict.
    primary_rows = []
    for regime in ("benign", "conflict"):
        g = final[final.regime == regime]
        cm = g[g.condition == "coordinate_mortality"].set_index("seed")
        nm = g[g.condition == "norm_matched"].set_index("seed")
        common = sorted(set(cm.index) & set(nm.index))
        result = exact_signflip_paired(cm.loc[common, "alignment_accuracy"], nm.loc[common, "alignment_accuracy"])
        primary_rows.append({
            "model": "Qwen3-8B",
            "regime": regime,
            "condition_A": "coordinate_mortality",
            "condition_B": "norm_matched",
            "metric": "alignment_accuracy",
            "followup_role": "primary_targeted",
            "minimum_attainable_two_sided_p": 2.0 / (2 ** result["n"]),
            **result,
        })
    adjusted = holm_adjust([row["p_exact"] for row in primary_rows])
    for row, p_adj in zip(primary_rows, adjusted):
        row["p_holm_across_two_regimes"] = p_adj
    pd.DataFrame(primary_rows).to_csv(args.out / "norm_matched_primary_exact_tests.csv", index=False)

    # Secondary paired comparisons.
    secondary_rows = []
    for regime in ("benign", "conflict"):
        g = final[final.regime == regime]
        cm = g[g.condition == "coordinate_mortality"].set_index("seed")
        nm = g[g.condition == "norm_matched"].set_index("seed")
        common = sorted(set(cm.index) & set(nm.index))
        for metric in ["alignment_margin_mean", "shift_nll_improvement", "relative_l2_drift"]:
            result = exact_signflip_paired(cm.loc[common, metric], nm.loc[common, metric])
            secondary_rows.append({
                "model": "Qwen3-8B",
                "regime": regime,
                "condition_A": "coordinate_mortality",
                "condition_B": "norm_matched",
                "metric": metric,
                "followup_role": "secondary_exploratory",
                **result,
            })
    pd.DataFrame(secondary_rows).to_csv(args.out / "norm_matched_secondary_exact_tests.csv", index=False)

    # Numerical verification of the matched norm.
    phase_target = target[target.phase > 0].copy()
    verification = pd.DataFrame([{
        "rows": len(phase_target),
        "max_abs_norm_match_error_after_bf16": phase_target["norm_match_abs_error_after_bf16"].max(),
        "max_relative_norm_match_error_after_bf16": phase_target["norm_match_relative_error_after_bf16"].max(),
        "max_computed_rounding_bound": phase_target["norm_match_rounding_bound"].max(),
        "min_raw_vs_post_direction_cosine": phase_target["norm_match_direction_cosine_raw_vs_post"].min(),
        "mean_counterfactual_cm_killed_fraction": phase_target["counterfactual_cm_killed_fraction"].mean(),
    }])
    verification.to_csv(args.out / "norm_match_numerical_verification.csv", index=False)

    # Behavioral judge comparisons if the targeted judge has been run.
    if args.original_judge.exists() and args.target_judge.exists():
        oj = pd.read_csv(args.original_judge)
        tj = pd.read_csv(args.target_judge)
        oj = oj[(oj.model == "Qwen3-8B") & (oj.phase == 6) & (oj.condition == "coordinate_mortality")]
        tj = tj[(tj.model == "Qwen3-8B") & (tj.phase == 6) & (tj.condition == "norm_matched")]
        judge_rows = []
        for regime in ("benign", "conflict"):
            a = oj[oj.regime == regime].set_index("seed")
            b = tj[tj.regime == regime].set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            for metric in ["judge_generated_mean", "judge_win_vs_rejected", "judge_gap_vs_rejected"]:
                result = exact_signflip_paired(a.loc[common, metric], b.loc[common, metric])
                judge_rows.append({
                    "model": "Qwen3-8B",
                    "regime": regime,
                    "condition_A": "coordinate_mortality",
                    "condition_B": "norm_matched",
                    "metric": metric,
                    "followup_role": "secondary_behavioral",
                    **result,
                })
        pd.DataFrame(judge_rows).to_csv(args.out / "norm_matched_behavior_exact_tests.csv", index=False)

    # Figures: phase trajectories and stability-plasticity.
    colors = {"coordinate_mortality": "C0", "norm_matched": "C3", "unconstrained": "C2"}
    for regime in ("benign", "conflict"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for condition in ("coordinate_mortality", "norm_matched", "unconstrained"):
            g = comparison[(comparison.regime == regime) & (comparison.condition == condition)]
            means = g.groupby("phase")["alignment_accuracy"].mean()
            ax.plot(means.index, means.values, marker="o", label=condition, color=colors[condition])
        ax.set_xlabel("Continual-learning phase")
        ax.set_ylabel("Held-out preference accuracy")
        ax.set_title(f"Norm-matched control | Qwen3-8B | {regime}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / f"norm_matched_alignment_Qwen3-8B_{regime}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for regime, marker in (("benign", "o"), ("conflict", "s")):
        for condition in ("coordinate_mortality", "norm_matched", "unconstrained"):
            g = final[(final.regime == regime) & (final.condition == condition)]
            ax.scatter(
                g["shift_nll_improvement"].mean(),
                g["alignment_accuracy_change"].mean(),
                label=f"{regime}:{condition}",
                marker=marker,
                s=70,
                color=colors[condition],
            )
    ax.axhline(0, linewidth=1, color="black", alpha=0.5)
    ax.axvline(0, linewidth=1, color="black", alpha=0.5)
    ax.set_xlabel("Held-out shift NLL improvement (plasticity)")
    ax.set_ylabel("Change in preference accuracy (stability)")
    ax.set_title("Coordinate selection versus norm-matched shrinkage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "norm_matched_stability_plasticity.pdf")
    plt.close(fig)

    print("NORM-MATCHED ANALYSIS COMPLETE")
    print(args.out / "norm_matched_primary_exact_tests.csv")
    print(args.out / "norm_matched_group_summaries.csv")


if __name__ == "__main__":
    main()
