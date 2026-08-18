#!/usr/bin/env python3
"""Deterministic analysis for the AAAI-27 experiment."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aaai27_core import load_config


def exact_signflip_paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    differences = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n == 0:
        return {"mean_difference": float("nan"), "p_exact": float("nan"), "n": 0}
    observed = abs(differences.mean())
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        statistics.append(abs(np.mean(differences * np.asarray(signs))))
    p_value = float(np.mean(np.asarray(statistics) >= observed - 1e-15))
    return {"mean_difference": float(differences.mean()), "p_exact": p_value, "n": int(n)}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    result = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return result.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        adjusted = min((m - rank) * p[index], 1.0)
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


_BOOTSTRAP_INDEX_CACHE: Dict[int, np.ndarray] = {}


def exact_bootstrap_mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # Enumerate every ordered n-out-of-n bootstrap sample. For the primary
    # seven-seed design this is 7^7 = 823,543 means, eliminating Monte Carlo
    # bootstrap noise.
    if n not in _BOOTSTRAP_INDEX_CACHE:
        _BOOTSTRAP_INDEX_CACHE[n] = np.indices((n,) * n, dtype=np.int16).reshape(n, -1).T
    grids = _BOOTSTRAP_INDEX_CACHE[n]
    means = x[grids].mean(axis=1)
    alpha = 1 - confidence
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def enrich(dataframe: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "regime", "condition"]
    frame = dataframe.sort_values(keys + ["phase"]).copy()
    for column in ["alignment_accuracy", "alignment_margin_mean", "shift_nll"]:
        baseline = frame.groupby(keys)[column].transform("first")
        if column == "shift_nll":
            frame["shift_nll_improvement"] = baseline - frame[column]
        else:
            frame[f"{column}_change"] = frame[column] - baseline
    frame["phase_alignment_accuracy_change"] = frame.groupby(keys)["alignment_accuracy"].diff()
    frame["phase_alignment_margin_change"] = frame.groupby(keys)["alignment_margin_mean"].diff()
    return frame


def validate_complete(frame: pd.DataFrame, config_path: Path, allow_incomplete: bool) -> None:
    config = load_config(config_path)
    expected = {
        (model.short_name, seed, regime, condition, phase)
        for model in config.models.values()
        for seed in model.seeds
        for regime in config.protocol.regimes
        for condition in config.protocol.conditions
        for phase in range(config.protocol.num_phases + 1)
    }
    observed = set(
        frame[["model", "seed", "regime", "condition", "phase"]]
        .itertuples(index=False, name=None)
    )
    missing = sorted(expected - observed)
    if missing and not allow_incomplete:
        sample = missing[:10]
        raise RuntimeError(
            f"Analysis is incomplete: {len(missing)} expected phase rows are missing. "
            f"First missing rows: {sample}. Re-run the same training command to resume."
        )


def model_role_map(config_path: Path) -> Dict[str, str]:
    config = load_config(config_path)
    return {model.short_name: model.inference_role for model in config.models.values()}


def group_summaries(final: pd.DataFrame, roles: Dict[str, str]) -> pd.DataFrame:
    metrics = [
        "alignment_accuracy_change",
        "alignment_margin_mean_change",
        "shift_nll_improvement",
        "relative_l2_drift",
        "retained_update_norm_ratio",
    ]
    rows = []
    for keys, group in final.groupby(["model", "regime", "condition"]):
        for metric in metrics:
            mean, lower, upper = exact_bootstrap_mean_ci(group[metric].to_numpy())
            rows.append(
                {
                    "model": keys[0],
                    "regime": keys[1],
                    "condition": keys[2],
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower_exact_bootstrap": lower,
                    "ci95_upper_exact_bootstrap": upper,
                    "seeds": int(group["seed"].nunique()),
                    "inference_role": roles[keys[0]],
                }
            )
    return pd.DataFrame(rows)


def paired_tests(
    final: pd.DataFrame,
    roles: Dict[str, str],
    minimum_primary_seeds: int,
) -> pd.DataFrame:
    metrics = [
        "alignment_accuracy",
        "alignment_margin_mean",
        "shift_nll_improvement",
        "relative_l2_drift",
    ]
    rows = []
    for (model, regime), group in final.groupby(["model", "regime"]):
        conditions = sorted(group["condition"].unique())
        for metric in metrics:
            metric_rows = []
            for left, right in itertools.combinations(conditions, 2):
                a = group[group.condition == left].set_index("seed")
                b = group[group.condition == right].set_index("seed")
                common = sorted(set(a.index) & set(b.index))
                result = exact_signflip_paired(
                    a.loc[common, metric].to_numpy(),
                    b.loc[common, metric].to_numpy(),
                )
                metric_rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "condition_A": left,
                        "condition_B": right,
                        "metric": metric,
                        "inference_role": roles[model],
                        "primary_confirmatory_test": bool(
                            roles[model] == "primary" and result["n"] >= minimum_primary_seeds
                        ),
                        "minimum_attainable_two_sided_p": (
                            2.0 / (2 ** result["n"]) if result["n"] else float("nan")
                        ),
                        **result,
                    }
                )
            adjusted = holm_adjust([row["p_exact"] for row in metric_rows])
            for row, p_adjusted in zip(metric_rows, adjusted):
                row["p_holm_within_model_regime_metric"] = p_adjusted
                rows.append(row)
    return pd.DataFrame(rows)


def judge_paired_tests(
    final_judge: pd.DataFrame,
    roles: Dict[str, str],
    minimum_primary_seeds: int,
) -> pd.DataFrame:
    metrics = [
        "judge_generated_mean",
        "judge_win_vs_rejected",
        "judge_gap_vs_rejected",
        "judge_gap_to_chosen",
    ]
    rows = []
    for (model, regime), group in final_judge.groupby(["model", "regime"]):
        conditions = sorted(group["condition"].unique())
        for metric in metrics:
            family = []
            for left, right in itertools.combinations(conditions, 2):
                a = group[group.condition == left].set_index("seed")
                b = group[group.condition == right].set_index("seed")
                common = sorted(set(a.index) & set(b.index))
                result = exact_signflip_paired(
                    a.loc[common, metric].to_numpy(),
                    b.loc[common, metric].to_numpy(),
                )
                family.append(
                    {
                        "model": model,
                        "regime": regime,
                        "condition_A": left,
                        "condition_B": right,
                        "metric": metric,
                        "inference_role": roles[model],
                        "primary_confirmatory_test": bool(
                            roles[model] == "primary"
                            and result["n"] >= minimum_primary_seeds
                        ),
                        "minimum_attainable_two_sided_p": (
                            2.0 / (2 ** result["n"])
                            if result["n"]
                            else float("nan")
                        ),
                        **result,
                    }
                )
            adjusted = holm_adjust([row["p_exact"] for row in family])
            for row, p_adjusted in zip(family, adjusted):
                row["p_holm_within_model_regime_metric"] = p_adjusted
                rows.append(row)
    return pd.DataFrame(rows)


def fit_ols(train_x: np.ndarray, train_y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.var(x) == 0:
        return float(np.mean(y)), 0.0
    design = np.column_stack([np.ones_like(x), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def direction_predictive_analysis(
    frame: pd.DataFrame,
    roles: Dict[str, str],
    minimum_primary_seeds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = frame[(frame.phase > 0) & (frame.condition == "unconstrained")].copy()
    features = {
        "direction_cosine": "raw_update_alignment_descent_cosine",
        "update_norm": "raw_update_norm",
    }
    rows = []
    for (model, regime), group in data.groupby(["model", "regime"]):
        seeds = sorted(group.seed.unique())
        for held_seed in seeds:
            train = group[group.seed != held_seed]
            test = group[group.seed == held_seed]
            target = "phase_alignment_margin_change"
            for label, feature in features.items():
                intercept, slope = fit_ols(train[feature].to_numpy(), train[target].to_numpy())
                prediction = intercept + slope * test[feature].to_numpy(dtype=float)
                actual = test[target].to_numpy(dtype=float)
                mask = np.isfinite(prediction) & np.isfinite(actual)
                rmse = float(np.sqrt(np.mean((prediction[mask] - actual[mask]) ** 2)))
                rows.append(
                    {
                        "model": model,
                        "regime": regime,
                        "held_out_seed": held_seed,
                        "predictor": label,
                        "rmse": rmse,
                        "train_intercept": intercept,
                        "train_slope": slope,
                        "test_points": int(mask.sum()),
                        "inference_role": roles[model],
                    }
                )
    per_seed = pd.DataFrame(rows)
    comparisons = []
    for (model, regime), group in per_seed.groupby(["model", "regime"]):
        direction = group[group.predictor == "direction_cosine"].set_index("held_out_seed")
        norm = group[group.predictor == "update_norm"].set_index("held_out_seed")
        common = sorted(set(direction.index) & set(norm.index))
        result = exact_signflip_paired(
            direction.loc[common, "rmse"].to_numpy(),
            norm.loc[common, "rmse"].to_numpy(),
        )
        comparisons.append(
            {
                "model": model,
                "regime": regime,
                "comparison": "direction_RMSE_minus_update_norm_RMSE",
                "inference_role": roles[model],
                "primary_confirmatory_test": bool(
                    roles[model] == "primary" and result["n"] >= minimum_primary_seeds
                ),
                "minimum_attainable_two_sided_p": (
                    2.0 / (2 ** result["n"]) if result["n"] else float("nan")
                ),
                **result,
            }
        )
    return per_seed, pd.DataFrame(comparisons)


def summarize_alignment_stage(
    alignment: pd.DataFrame,
    roles: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = alignment.copy()
    frame["alignment_stage_accuracy_gain"] = (
        frame["post_alignment_alignment_accuracy"]
        - frame["base_alignment_alignment_accuracy"]
    )
    frame["alignment_stage_margin_gain"] = (
        frame["post_alignment_alignment_margin_mean"]
        - frame["base_alignment_alignment_margin_mean"]
    )
    rows = []
    for model, group in frame.groupby("model"):
        for metric in (
            "base_alignment_alignment_accuracy",
            "post_alignment_alignment_accuracy",
            "alignment_stage_accuracy_gain",
            "base_alignment_alignment_margin_mean",
            "post_alignment_alignment_margin_mean",
            "alignment_stage_margin_gain",
        ):
            mean, lower, upper = exact_bootstrap_mean_ci(group[metric].to_numpy())
            rows.append(
                {
                    "model": model,
                    "inference_role": roles[model],
                    "metric": metric,
                    "mean": mean,
                    "ci95_lower_exact_bootstrap": lower,
                    "ci95_upper_exact_bootstrap": upper,
                    "seeds": int(group["seed"].nunique()),
                }
            )
    return frame, pd.DataFrame(rows)


def make_figures(frame: pd.DataFrame, final: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for (model, regime), group in frame.groupby(["model", "regime"]):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for condition, subset in group.groupby("condition"):
            summary = subset.groupby("phase")["alignment_accuracy"].agg(["mean", "std"]).reset_index()
            std = summary["std"].fillna(0)
            axis.plot(summary.phase, summary["mean"], marker="o", label=condition)
            axis.fill_between(summary.phase, summary["mean"] - std, summary["mean"] + std, alpha=0.15)
        axis.set_xlabel("Continual-learning phase")
        axis.set_ylabel("Held-out preference accuracy")
        axis.set_title(f"{model} | {regime} shift")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"alignment_{model}_{regime}.pdf")
        plt.close(figure)

    for model, group in final.groupby("model"):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for (regime, condition), subset in group.groupby(["regime", "condition"]):
            axis.scatter(
                subset["shift_nll_improvement"].mean(),
                subset["alignment_accuracy_change"].mean(),
                s=70,
                label=f"{regime}:{condition}",
            )
        axis.axhline(0, linewidth=1)
        axis.axvline(0, linewidth=1)
        axis.set_xlabel("Held-out shift NLL improvement (plasticity)")
        axis.set_ylabel("Change in preference accuracy (stability)")
        axis.set_title(f"Stability-plasticity frontier | {model}")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output / f"stability_plasticity_{model}.pdf")
        plt.close(figure)

    unconstrained = frame[(frame.phase > 0) & (frame.condition == "unconstrained")]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime, subset in unconstrained.groupby("regime"):
        axis.scatter(
            subset["raw_update_alignment_descent_cosine"],
            subset["phase_alignment_margin_change"],
            alpha=0.5,
            s=22,
            label=regime,
        )
    axis.axhline(0, linewidth=1)
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Raw update cosine with alignment descent")
    axis.set_ylabel("Observed phase change in preference margin")
    axis.set_title("Update direction versus alignment change")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "direction_vs_alignment_change.pdf")
    plt.close(figure)

    mortality = frame[(frame.phase > 0) & (frame.condition == "coordinate_mortality")]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for regime, subset in mortality.groupby("regime"):
        summary = subset.groupby("phase")["coordinate_killed_fraction"].agg(["mean", "std"]).reset_index()
        std = summary["std"].fillna(0)
        axis.plot(summary.phase, summary["mean"], marker="o", label=regime)
        axis.fill_between(summary.phase, summary["mean"] - std, summary["mean"] + std, alpha=0.15)
    axis.set_xlabel("Phase")
    axis.set_ylabel("Fraction of raw update coordinates deleted")
    axis.set_title("Coordinate mortality intervention strength")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "coordinate_mortality_fraction.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("pci_h100_outputs"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--allow-incomplete", action="store_true")
    arguments = parser.parse_args()

    results = arguments.root / "results"
    metrics_path = results / "all_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    frame = pd.read_csv(metrics_path)
    validate_complete(frame, arguments.config, arguments.allow_incomplete)
    frame = enrich(frame)
    frame.to_csv(results / "all_metrics_enriched.csv", index=False)

    config = load_config(arguments.config)
    final = frame[frame.phase == config.protocol.num_phases].copy()
    final.to_csv(results / "final_stability_plasticity.csv", index=False)

    roles = model_role_map(arguments.config)
    alignment_path = results / "alignment_stage_metrics.csv"
    if alignment_path.exists():
        alignment_raw = pd.read_csv(alignment_path)
        alignment_enriched, alignment_summary = summarize_alignment_stage(
            alignment_raw, roles
        )
        alignment_enriched.to_csv(
            results / "alignment_stage_metrics_enriched.csv", index=False
        )
        alignment_summary.to_csv(
            results / "alignment_stage_summary_exact_bootstrap.csv", index=False
        )

    summaries = group_summaries(final, roles)
    summaries.to_csv(results / "final_group_summaries_exact_bootstrap.csv", index=False)
    tests = paired_tests(final, roles, config.protocol.minimum_exact_test_seeds)
    tests.to_csv(results / "exact_paired_randomization_tests.csv", index=False)
    predictive, predictive_test = direction_predictive_analysis(
        frame, roles, config.protocol.minimum_exact_test_seeds
    )
    predictive.to_csv(results / "direction_prediction_loso_by_seed.csv", index=False)
    predictive_test.to_csv(results / "direction_vs_magnitude_predictive_tests.csv", index=False)

    judge_path = results / "behavior_judge_aggregate.csv"
    if judge_path.exists():
        validity_path = results / "judge_validity.json"
        if not validity_path.exists():
            raise RuntimeError("Behavior scores exist without judge_validity.json")
        import json

        validity = json.loads(validity_path.read_text())
        if not bool(validity.get("calibration_passed", False)):
            raise RuntimeError("Behavior judge did not pass the preregistered validity gate")
        judge = pd.read_csv(judge_path)
        final_judge = judge[judge.checkpoint_kind == "final"].drop(columns=["checkpoint_kind"])
        merged = final.merge(
            final_judge,
            on=["model", "seed", "regime", "condition", "phase"],
            how="left",
            validate="one_to_one",
        )
        if merged["judge_generated_mean"].isna().any():
            raise RuntimeError("Some final checkpoints are missing independent judge scores")
        merged.to_csv(results / "final_stability_plasticity_with_judge.csv", index=False)
        judge_tests = judge_paired_tests(
            final_judge, roles, config.protocol.minimum_exact_test_seeds
        )
        judge_tests.to_csv(results / "behavior_judge_exact_paired_tests.csv", index=False)

    make_figures(frame, final, results / "figures")
    print(f"ANALYSIS COMPLETE: {results}")


if __name__ == "__main__":
    main()
