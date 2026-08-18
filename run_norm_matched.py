#!/usr/bin/env python3
"""Targeted AAAI-27 follow-up: norm-matched shrinkage control.

Scientific purpose
------------------
Coordinate Mortality (CM) changes two things at once: it removes coordinates
with g_i u_i > 0 and consequently shrinks the update norm.  This follow-up
isolates those effects.  At each continual-learning phase, after ordinary shift
training produces raw update u and the held-out alignment-control gradient g,
we compute the *counterfactual* CM update

    v_CM,i = 0      if g_i u_i > 0
             u_i    otherwise

and derive, without a tuned hyperparameter,

    r = ||v_CM||_2 / ||u||_2.

The Norm-Matched (NM) control then applies

    v_NM = r u.

Thus NM has the same FP32 update norm as CM would have at exactly the same model
state, while preserving the raw update direction and deleting no coordinates.
If CM outperforms NM, simple update shrinkage is insufficient to explain CM.

This script is intentionally limited to Qwen3-8B, the seven preregistered
primary seeds, the two existing shift regimes, and the inherited six-phase
protocol.  It reuses the already-trained preference checkpoints from
pci_h100_outputs and never retrains the alignment stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
import torch

from aaai27_core import (
    DTYPE,
    PairDataset,
    atomic_json,
    cosine,
    derived_seed,
    drift_metrics,
    ensure_dirs,
    evaluate_alignment,
    evaluate_shift_nll,
    full_alignment_gradient,
    generate_behavior_sample,
    get_tokenizer,
    load_asset_manifest,
    load_config,
    load_lora_model,
    load_lora_state,
    load_partitions,
    lora_state_cpu,
    lora_vector,
    make_pair_loader,
    make_shift_loader,
    release_cuda,
    require_h100,
    restore_lora_state,
    save_lora_state,
    set_seed,
    state_vector,
    train_shift_phase,
    write_lora_vector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("aaai27.norm_matched")
DEVICE = require_h100()
CONDITION = "norm_matched"


def checkpoint_path(directory: Path, phase: int) -> Path:
    return directory / f"phase_{phase}" / "lora.safetensors"


def metric_path(directory: Path, phase: int) -> Path:
    return directory / f"phase_{phase}" / "metrics.json"


def source_aligned_path(source_root: Path, model_name: str, seed: int) -> Path:
    return source_root / "aligned" / model_name / f"seed_{seed}" / "lora.safetensors"


def branch_dir(root_dirs: Mapping[str, Path], model_name: str, seed: int, regime: str) -> Path:
    return root_dirs["runs"] / model_name / f"seed_{seed}" / regime / CONDITION


def evaluate_phase(model, test_loader, shift_eval_loader, anchor_vector, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(metadata)
    row.update(evaluate_alignment(model, test_loader, DEVICE))
    row.update(evaluate_shift_nll(model, shift_eval_loader, DEVICE))
    row.update(drift_metrics(model, anchor_vector))
    return row


@torch.no_grad()
def apply_norm_matched_control(
    model,
    pre_vector: torch.Tensor,
    alignment_gradient: torch.Tensor,
) -> Dict[str, float]:
    """Apply norm-matched shrinkage, with no fitted intervention strength.

    The scalar r is derived from the norm of the counterfactual Coordinate
    Mortality update at the *same* state and phase.  This provides the clean
    control needed to separate coordinate selection from simple norm shrinkage.
    """
    raw_post = lora_vector(model)
    raw_update = raw_post - pre_vector
    raw_norm = raw_update.norm()
    raw_harm = torch.dot(alignment_gradient, raw_update)

    coordinate_harm = alignment_gradient * raw_update
    counterfactual_kill = coordinate_harm > 0
    counterfactual_cm = raw_update.masked_fill(counterfactual_kill, 0.0)
    counterfactual_cm_norm = counterfactual_cm.norm()

    if raw_norm.item() == 0.0:
        ratio = torch.ones((), device=raw_update.device, dtype=raw_update.dtype)
    else:
        ratio = counterfactual_cm_norm / raw_norm

    controlled = raw_update * ratio
    constructed_norm = controlled.norm()
    # In FP32 this must match by construction, up to FP32 roundoff.
    fp32_norm_error = torch.abs(constructed_norm - counterfactual_cm_norm)
    fp32_bound = 16 * torch.finfo(torch.float32).eps * max(
        raw_norm.item(), counterfactual_cm_norm.item(), 1.0
    )
    if fp32_norm_error.item() > fp32_bound:
        raise RuntimeError(
            "Norm-matched construction failed in FP32: "
            f"error={fp32_norm_error.item()} bound={fp32_bound}"
        )

    constructed_harm = torch.dot(alignment_gradient, controlled)
    post_vector = pre_vector + controlled
    write_lora_vector(model, post_vector)

    # Re-read BF16 state.  Since pre_vector is an exact FP32 representation of
    # the pre-existing BF16 state, only post_vector is newly quantized here.
    actual_controlled = lora_vector(model) - pre_vector
    actual_norm = actual_controlled.norm()
    post_harm = torch.dot(alignment_gradient, actual_controlled)

    unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
    bf16_vector_error_bound = unit_roundoff * post_vector.norm().item()
    fp32_readback_bound = 16 * torch.finfo(torch.float32).eps * max(
        post_vector.norm().item(), actual_norm.item(), 1.0
    )
    norm_match_tolerance = bf16_vector_error_bound + fp32_readback_bound
    norm_match_abs_error = abs(actual_norm.item() - counterfactual_cm_norm.item())
    if norm_match_abs_error > norm_match_tolerance:
        raise RuntimeError(
            "Stored BF16 norm differs from counterfactual CM norm beyond the "
            "computed rounding bound: "
            f"error={norm_match_abs_error} bound={norm_match_tolerance}"
        )

    intervention = actual_controlled - raw_update
    positive_risk = coordinate_harm.clamp_min(0).sum()
    negative_risk_abs = (-coordinate_harm.clamp_max(0)).sum()
    return {
        "raw_first_order_alignment_change": raw_harm.item(),
        "constructed_first_order_alignment_change": constructed_harm.item(),
        "post_first_order_alignment_change": post_harm.item(),
        "raw_update_alignment_descent_cosine": cosine(raw_update, -alignment_gradient),
        "post_update_alignment_descent_cosine": cosine(actual_controlled, -alignment_gradient),
        "raw_update_norm": raw_norm.item(),
        "post_update_norm": actual_norm.item(),
        "retained_update_norm_ratio": (actual_norm / raw_norm.clamp_min(1e-20)).item(),
        "intervention_norm": intervention.norm().item(),
        "coordinate_killed_fraction": 0.0,
        "positive_coordinate_risk_fraction": counterfactual_kill.float().mean().item(),
        "positive_coordinate_risk_sum": positive_risk.item(),
        "negative_coordinate_risk_abs_sum": negative_risk_abs.item(),
        "global_projection_activated": 0.0,
        "counterfactual_cm_update_norm": counterfactual_cm_norm.item(),
        "counterfactual_cm_retained_norm_ratio": ratio.item(),
        "counterfactual_cm_killed_fraction": counterfactual_kill.float().mean().item(),
        "norm_match_abs_error_after_bf16": norm_match_abs_error,
        "norm_match_relative_error_after_bf16": (
            norm_match_abs_error / max(counterfactual_cm_norm.item(), 1e-30)
        ),
        "norm_match_rounding_bound": norm_match_tolerance,
        "norm_match_direction_cosine_raw_vs_post": cosine(raw_update, actual_controlled),
    }


def consolidate(root_dirs: Mapping[str, Path]) -> None:
    rows = [json.loads(path.read_text()) for path in sorted(root_dirs["runs"].glob("**/metrics.json"))]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    keys = ["model", "seed", "regime", "condition", "phase"]
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(root_dirs["results"] / "norm_matched_metrics.csv", index=False)
    LOG.info("Consolidated %d norm-matched phase rows", len(frame))


def run_branch(
    model,
    tokenizer,
    partitions,
    prepared_control: PairDataset,
    prepared_test: PairDataset,
    protocol,
    runtime,
    seed: int,
    regime: str,
    aligned_state: Mapping[str, torch.Tensor],
    root_dirs: Mapping[str, Path],
) -> None:
    directory = branch_dir(root_dirs, runtime.short_name, seed, regime)
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "DONE.json"
    anchor_vector = state_vector(aligned_state, model, DEVICE)
    test_loader = make_pair_loader(
        prepared_test, tokenizer, runtime.eval_microbatch, shuffle=False, seed=seed
    )
    control_loader = make_pair_loader(
        prepared_control, tokenizer, runtime.control_microbatch, shuffle=False, seed=seed
    )

    latest = -1
    for phase in range(protocol.num_phases, -1, -1):
        if checkpoint_path(directory, phase).exists() and metric_path(directory, phase).exists():
            latest = phase
            break

    if done_path.exists() and latest == protocol.num_phases:
        LOG.info("SKIP complete %s seed=%d %s %s", runtime.short_name, seed, regime, CONDITION)
        load_lora_state(model, checkpoint_path(directory, protocol.num_phases))
        generation_path = root_dirs["generations"] / (
            f"{runtime.short_name}_seed{seed}_{regime}_{CONDITION}_final.jsonl"
        )
        generate_behavior_sample(
            model,
            tokenizer,
            partitions.behavior_test,
            runtime.generation_microbatch,
            seed,
            generation_path,
            {
                "model": runtime.short_name,
                "seed": seed,
                "regime": regime,
                "condition": CONDITION,
                "phase": protocol.num_phases,
                "checkpoint_kind": "final",
            },
            DEVICE,
        )
        return

    if latest >= 0:
        LOG.info("RESUME %s seed=%d %s %s at phase=%d", runtime.short_name, seed, regime, CONDITION, latest)
        load_lora_state(model, checkpoint_path(directory, latest))
    else:
        restore_lora_state(model, aligned_state, DEVICE)
        latest = 0
        shift_eval_loader = make_shift_loader(
            regime, partitions, tokenizer, protocol, runtime, seed, phase=0, eval_mode=True
        )
        row0 = evaluate_phase(
            model,
            test_loader,
            shift_eval_loader,
            anchor_vector,
            {
                "model": runtime.short_name,
                "seed": seed,
                "regime": regime,
                "condition": CONDITION,
                "phase": 0,
                "cumulative_target_tokens": 0,
            },
        )
        save_lora_state(model, checkpoint_path(directory, 0))
        atomic_json(row0, metric_path(directory, 0))

    for phase in range(latest + 1, protocol.num_phases + 1):
        LOG.info(
            "RUN TARGETED model=%s seed=%d regime=%s condition=%s phase=%d/%d",
            runtime.short_name,
            seed,
            regime,
            CONDITION,
            phase,
            protocol.num_phases,
        )
        # Exactly the same phase-specific stream used by the original experiment.
        set_seed(derived_seed("shift-phase", runtime.short_name, seed, regime, phase))
        pre_vector = lora_vector(model).clone()
        alignment_gradient, control_loss = full_alignment_gradient(model, control_loader, DEVICE)
        shift_train_loader = make_shift_loader(
            regime, partitions, tokenizer, protocol, runtime, seed, phase, eval_mode=False
        )
        training = train_shift_phase(model, shift_train_loader, protocol, runtime, DEVICE)
        intervention = apply_norm_matched_control(model, pre_vector, alignment_gradient)
        shift_eval_loader = make_shift_loader(
            regime, partitions, tokenizer, protocol, runtime, seed, phase, eval_mode=True
        )
        row = evaluate_phase(
            model,
            test_loader,
            shift_eval_loader,
            anchor_vector,
            {
                "model": runtime.short_name,
                "seed": seed,
                "regime": regime,
                "condition": CONDITION,
                "phase": phase,
                "cumulative_target_tokens": phase * protocol.tokens_per_phase,
                "control_alignment_loss_pre": control_loss,
                **training,
                **intervention,
            },
        )
        save_lora_state(model, checkpoint_path(directory, phase))
        atomic_json(row, metric_path(directory, phase))
        del pre_vector, alignment_gradient
        release_cuda()

    atomic_json(
        {
            "status": "complete",
            "followup": "norm_matched_shrinkage",
            "model": runtime.short_name,
            "seed": seed,
            "regime": regime,
            "condition": CONDITION,
            "phases": protocol.num_phases,
        },
        done_path,
    )
    generation_path = root_dirs["generations"] / (
        f"{runtime.short_name}_seed{seed}_{regime}_{CONDITION}_final.jsonl"
    )
    generate_behavior_sample(
        model,
        tokenizer,
        partitions.behavior_test,
        runtime.generation_microbatch,
        seed,
        generation_path,
        {
            "model": runtime.short_name,
            "seed": seed,
            "regime": regime,
            "condition": CONDITION,
            "phase": protocol.num_phases,
            "checkpoint_kind": "final",
        },
        DEVICE,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.json"))
    p.add_argument("--assets", type=Path, default=Path("asset_manifest.json"))
    p.add_argument("--source-root", type=Path, default=Path("pci_h100_outputs"))
    p.add_argument("--root", type=Path, default=Path("norm_matched_outputs"))
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assets = load_asset_manifest(args.assets)
    runtime = config.models["8b"]
    protocol = config.protocol
    seeds: Sequence[int] = tuple(args.seeds) if args.seeds else runtime.seeds
    unknown = set(seeds) - set(runtime.seeds)
    if unknown:
        raise ValueError(f"Unregistered 8B seeds: {sorted(unknown)}; allowed={list(runtime.seeds)}")

    source_root = args.source_root.resolve()
    target_dirs = ensure_dirs(args.root)
    source_partition_manifest = source_root / "manifests" / "partition_manifest.json"
    if not source_partition_manifest.exists():
        raise FileNotFoundError(f"Missing original partition manifest: {source_partition_manifest}")

    partitions = load_partitions(protocol, assets, target_dirs["manifests"] / "partition_manifest.json")
    original_manifest = json.loads(source_partition_manifest.read_text())
    target_manifest = json.loads((target_dirs["manifests"] / "partition_manifest.json").read_text())
    if original_manifest != target_manifest:
        raise RuntimeError("Targeted follow-up did not reproduce the original deterministic data partition")

    revision = assets["models"][runtime.model_id]
    tokenizer = get_tokenizer(runtime.model_id, revision)
    prepared_control = PairDataset(partitions.align_control, tokenizer, protocol.max_length)
    prepared_test = PairDataset(partitions.test, tokenizer, protocol.max_length)

    followup_manifest = {
        "followup": "norm_matched_shrinkage",
        "scientific_question": (
            "Does Coordinate Mortality outperform an update with the same locally-derived "
            "L2 norm but unchanged raw direction?"
        ),
        "definition": "r=||CM(g,u)||/||u||; v_NM=r*u",
        "tuned_intervention_hyperparameters": 0,
        "model": runtime.short_name,
        "seeds": list(seeds),
        "regimes": list(protocol.regimes),
        "phases": protocol.num_phases,
        "tokens_per_phase": protocol.tokens_per_phase,
        "source_root": str(source_root),
        "model_revision": revision,
        "asset_manifest": assets,
        "original_partition_manifest": original_manifest,
        "started_unix": time.time(),
    }
    manifest_path = target_dirs["root"] / "FOLLOWUP_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        stable_keys = [
            "followup", "scientific_question", "definition", "tuned_intervention_hyperparameters",
            "model", "seeds", "regimes", "phases", "tokens_per_phase", "source_root",
            "model_revision", "asset_manifest", "original_partition_manifest",
        ]
        if any(existing.get(k) != followup_manifest.get(k) for k in stable_keys):
            raise RuntimeError("Existing norm_matched_outputs was created with a different follow-up protocol")
    else:
        atomic_json(followup_manifest, manifest_path)

    for seed in seeds:
        set_seed(seed)
        model, model_tokenizer = load_lora_model(runtime, protocol, revision, DEVICE)
        try:
            aligned_path = source_aligned_path(source_root, runtime.short_name, seed)
            if not aligned_path.exists():
                raise FileNotFoundError(
                    f"Missing original aligned checkpoint for seed {seed}: {aligned_path}. "
                    "Do not retrain it; restore the completed original experiment first."
                )
            load_lora_state(model, aligned_path)
            aligned_state = lora_state_cpu(model)
            for regime in protocol.regimes:
                restore_lora_state(model, aligned_state, DEVICE)
                run_branch(
                    model,
                    model_tokenizer,
                    partitions,
                    prepared_control,
                    prepared_test,
                    protocol,
                    runtime,
                    seed,
                    regime,
                    aligned_state,
                    target_dirs,
                )
                release_cuda()
            consolidate(target_dirs)
        finally:
            del model, model_tokenizer
            release_cuda()

    consolidate(target_dirs)
    LOG.info("NORM-MATCHED FOLLOW-UP COMPLETE")
    LOG.info("Next: bash run_norm_matched_judge.sh, then bash run_norm_matched_analysis.sh")


if __name__ == "__main__":
    main()
