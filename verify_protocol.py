#!/usr/bin/env python3
"""Pure-stdlib validation and exact compute accounting for the study design."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist


def wilson_required_n(confidence: float, half_width: float) -> int:
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    for n in range(1, 10_000_000):
        p = 0.5
        denominator = 1.0 + z * z / n
        half = (
            z
            * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
            / denominator
        )
        if half <= half_width:
            return n
    raise RuntimeError("Wilson sample-size search did not converge")


def minimum_exact_seeds(condition_count: int, family_alpha: float = 0.05) -> int:
    comparisons = math.comb(condition_count, 2)
    n = 1
    while comparisons * 2.0 / (2**n) >= family_alpha:
        n += 1
    return n


def main() -> None:
    path = Path("config.json")
    config = json.loads(path.read_text())
    protocol = config["protocol"]
    conditions = protocol["conditions"]
    regimes = protocol["regimes"]

    if set(conditions) != {
        "unconstrained",
        "global_projection",
        "coordinate_mortality",
    }:
        raise RuntimeError("Unexpected condition set")
    if set(regimes) != {"benign", "conflict"}:
        raise RuntimeError("Unexpected regime set")
    if protocol["tokens_per_phase"] % protocol["max_length"]:
        raise RuntimeError("tokens_per_phase must be divisible by max_length")

    required_primary = minimum_exact_seeds(len(conditions))
    primary_models = []
    rows = []
    total_shift_tokens = 0
    for key, model in config["models"].items():
        seeds = [int(x) for x in model["seeds"]]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"Duplicate seeds for {key}")
        if any(int(model[name]) <= 0 for name in (
            "pair_microbatch",
            "pair_grad_accum",
            "shift_microbatch",
            "shift_grad_accum",
            "eval_microbatch",
            "control_microbatch",
            "generation_microbatch",
        )):
            raise RuntimeError(f"Non-positive runtime setting for {key}")
        role = model["inference_role"]
        if role == "primary":
            primary_models.append(key)
            if len(seeds) < required_primary:
                raise RuntimeError(
                    f"Primary {key} has {len(seeds)} seeds; requires {required_primary}"
                )
        elif role != "scale_replication":
            raise RuntimeError(f"Unknown inference_role for {key}: {role}")

        branches = len(seeds) * len(regimes) * len(conditions)
        phase_runs = branches * int(protocol["num_phases"])
        tokens = phase_runs * int(protocol["tokens_per_phase"])
        total_shift_tokens += tokens
        rows.append(
            {
                "model": key,
                "role": role,
                "seeds": len(seeds),
                "branches": branches,
                "trained_phases": phase_runs,
                "exact_shift_target_tokens": tokens,
            }
        )

    if len(primary_models) != 1:
        raise RuntimeError("Exactly one primary model is required")

    behavior_n = wilson_required_n(
        float(protocol["behavior_confidence"]),
        float(protocol["behavior_half_width"]),
    )
    report = {
        "alignment_partition_pairs": int(protocol["tokens_per_phase"])
        // int(protocol["max_length"]),
        "behavior_sample_required_worst_case": behavior_n,
        "minimum_primary_exact_test_seeds": required_primary,
        "models": rows,
        "total_exact_shift_target_tokens": total_shift_tokens,
        "note": (
            "This count excludes alignment training, full control-gradient passes, "
            "evaluation, and generation."
        ),
    }
    print(json.dumps(report, indent=2))
    print("PROTOCOL CHECK PASSED")


if __name__ == "__main__":
    main()
