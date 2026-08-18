#!/usr/bin/env python3
"""Run the complete checkpointed H100 PCIe experiment.

The three conditions are parameter-free at intervention time:
  * unconstrained: keep the raw continual-learning update;
  * global_projection: Euclidean projection onto g^T v <= 0;
  * coordinate_mortality: delete coordinates with g_i u_i > 0.

All training, evaluation, gradients, update controls, drift calculations, and
generation execute on the H100. Tokenization and file I/O remain CPU-side.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
import torch

from aaai27_core import (
    Config,
    DataPartitions,
    ModelRuntime,
    PairDataset,
    Protocol,
    apply_update_control,
    atomic_json,
    drift_metrics,
    derived_seed,
    ensure_dirs,
    environment_manifest,
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
    train_preference_alignment,
    train_shift_phase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("aaai27.experiment")
DEVICE = require_h100()


@dataclass
class PreparedData:
    align_train: PairDataset
    align_control: PairDataset
    test: PairDataset
    tokenizer_summary: Dict[str, Any]


def aligned_dir(dirs: Mapping[str, Path], runtime: ModelRuntime, seed: int) -> Path:
    return dirs["aligned"] / runtime.short_name / f"seed_{seed}"


def branch_dir(
    dirs: Mapping[str, Path],
    runtime: ModelRuntime,
    seed: int,
    regime: str,
    condition: str,
) -> Path:
    return dirs["runs"] / runtime.short_name / f"seed_{seed}" / regime / condition


def checkpoint_path(directory: Path, phase: int) -> Path:
    return directory / f"phase_{phase}" / "lora.safetensors"


def metric_path(directory: Path, phase: int) -> Path:
    return directory / f"phase_{phase}" / "metrics.json"


def write_metric(row: Dict[str, Any], path: Path) -> None:
    atomic_json(row, path)


def prepare_model_data(
    runtime: ModelRuntime,
    protocol: Protocol,
    partitions: DataPartitions,
    model_revision: str,
) -> Tuple[Any, PreparedData]:
    tokenizer = get_tokenizer(runtime.model_id, model_revision)
    align_train = PairDataset(partitions.align_train, tokenizer, protocol.max_length)
    align_control = PairDataset(partitions.align_control, tokenizer, protocol.max_length)
    test = PairDataset(partitions.test, tokenizer, protocol.max_length)
    summary = {
        "alignment_train": align_train.truncation_summary(),
        "alignment_control": align_control.truncation_summary(),
        "official_test": test.truncation_summary(),
    }
    return tokenizer, PreparedData(align_train, align_control, test, summary)


def evaluate_phase(
    model,
    test_loader,
    shift_eval_loader,
    anchor_vector: torch.Tensor,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    row = dict(metadata)
    row.update(evaluate_alignment(model, test_loader, DEVICE))
    row.update(evaluate_shift_nll(model, shift_eval_loader, DEVICE))
    row.update(drift_metrics(model, anchor_vector))
    return row


def train_or_load_alignment(
    model,
    tokenizer,
    prepared: PreparedData,
    protocol: Protocol,
    runtime: ModelRuntime,
    seed: int,
    dirs: Mapping[str, Path],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    directory = aligned_dir(dirs, runtime, seed)
    state_path = directory / "lora.safetensors"
    metrics_path = directory / "metrics.json"
    train_loader = make_pair_loader(
        prepared.align_train,
        tokenizer,
        runtime.pair_microbatch,
        shuffle=True,
        seed=seed,
    )
    test_loader = make_pair_loader(
        prepared.test,
        tokenizer,
        runtime.eval_microbatch,
        shuffle=False,
        seed=seed,
    )

    if state_path.exists() and metrics_path.exists():
        LOG.info("Loading aligned checkpoint %s seed=%d", runtime.short_name, seed)
        load_lora_state(model, state_path)
        return lora_state_cpu(model), json.loads(metrics_path.read_text())

    set_seed(seed)
    base_metrics = evaluate_alignment(model, test_loader, DEVICE)
    train_metrics = train_preference_alignment(
        model,
        train_loader,
        protocol,
        runtime.pair_grad_accum,
        DEVICE,
    )
    aligned_metrics = evaluate_alignment(model, test_loader, DEVICE)
    save_lora_state(model, state_path)
    metrics = {
        "model": runtime.short_name,
        "seed": seed,
        "base_alignment": base_metrics,
        "post_alignment": aligned_metrics,
        **train_metrics,
    }
    atomic_json(metrics, metrics_path)
    return lora_state_cpu(model), metrics


def run_branch(
    model,
    tokenizer,
    prepared: PreparedData,
    partitions: DataPartitions,
    protocol: Protocol,
    runtime: ModelRuntime,
    seed: int,
    regime: str,
    condition: str,
    aligned_state: Dict[str, torch.Tensor],
    dirs: Mapping[str, Path],
) -> None:
    directory = branch_dir(dirs, runtime, seed, regime, condition)
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "DONE.json"
    anchor_vector = state_vector(aligned_state, model, DEVICE)
    test_loader = make_pair_loader(
        prepared.test,
        tokenizer,
        runtime.eval_microbatch,
        shuffle=False,
        seed=seed,
    )
    control_loader = make_pair_loader(
        prepared.align_control,
        tokenizer,
        runtime.control_microbatch,
        shuffle=False,
        seed=seed,
    )

    latest = -1
    for phase in range(protocol.num_phases, -1, -1):
        if checkpoint_path(directory, phase).exists() and metric_path(directory, phase).exists():
            latest = phase
            break

    if done_path.exists() and latest == protocol.num_phases:
        LOG.info("SKIP complete %s seed=%d %s %s", runtime.short_name, seed, regime, condition)
        load_lora_state(model, checkpoint_path(directory, protocol.num_phases))
        generation_path = dirs["generations"] / (
            f"{runtime.short_name}_seed{seed}_{regime}_{condition}_final.jsonl"
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
                "condition": condition,
                "phase": protocol.num_phases,
                "checkpoint_kind": "final",
            },
            DEVICE,
        )
        return

    if latest >= 0:
        LOG.info(
            "RESUME %s seed=%d %s %s at phase=%d",
            runtime.short_name,
            seed,
            regime,
            condition,
            latest,
        )
        load_lora_state(model, checkpoint_path(directory, latest))
    else:
        restore_lora_state(model, aligned_state, DEVICE)
        latest = 0
        shift_eval_loader = make_shift_loader(
            regime,
            partitions,
            tokenizer,
            protocol,
            runtime,
            seed,
            phase=0,
            eval_mode=True,
        )
        phase_zero = evaluate_phase(
            model,
            test_loader,
            shift_eval_loader,
            anchor_vector,
            {
                "model": runtime.short_name,
                "seed": seed,
                "regime": regime,
                "condition": condition,
                "phase": 0,
                "cumulative_target_tokens": 0,
            },
        )
        save_lora_state(model, checkpoint_path(directory, 0))
        write_metric(phase_zero, metric_path(directory, 0))

    for phase in range(latest + 1, protocol.num_phases + 1):
        LOG.info(
            "RUN model=%s seed=%d regime=%s condition=%s phase=%d/%d",
            runtime.short_name,
            seed,
            regime,
            condition,
            phase,
            protocol.num_phases,
        )
        # Paired conditions receive the same phase-specific stochastic stream
        # (dropout and any CUDA sampling) without relying on hand-picked seed
        # arithmetic. DataLoader order is independently fixed in make_shift_loader.
        set_seed(derived_seed("shift-phase", runtime.short_name, seed, regime, phase))
        pre_vector = lora_vector(model).clone()
        alignment_gradient, control_loss = full_alignment_gradient(model, control_loader, DEVICE)
        shift_train_loader = make_shift_loader(
            regime,
            partitions,
            tokenizer,
            protocol,
            runtime,
            seed,
            phase,
            eval_mode=False,
        )
        training = train_shift_phase(
            model,
            shift_train_loader,
            protocol,
            runtime,
            DEVICE,
        )
        intervention = apply_update_control(
            model,
            pre_vector,
            alignment_gradient,
            condition,
        )
        shift_eval_loader = make_shift_loader(
            regime,
            partitions,
            tokenizer,
            protocol,
            runtime,
            seed,
            phase,
            eval_mode=True,
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
                "condition": condition,
                "phase": phase,
                "cumulative_target_tokens": phase * protocol.tokens_per_phase,
                "control_alignment_loss_pre": control_loss,
                **training,
                **intervention,
            },
        )
        save_lora_state(model, checkpoint_path(directory, phase))
        write_metric(row, metric_path(directory, phase))
        del pre_vector, alignment_gradient
        release_cuda()

    atomic_json(
        {
            "status": "complete",
            "model": runtime.short_name,
            "seed": seed,
            "regime": regime,
            "condition": condition,
            "phases": protocol.num_phases,
        },
        done_path,
    )
    generation_path = dirs["generations"] / (
        f"{runtime.short_name}_seed{seed}_{regime}_{condition}_final.jsonl"
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
            "condition": condition,
            "phase": protocol.num_phases,
            "checkpoint_kind": "final",
        },
        DEVICE,
    )


def run_model_seed(
    runtime: ModelRuntime,
    protocol: Protocol,
    prepared: PreparedData,
    partitions: DataPartitions,
    model_revision: str,
    seed: int,
    dirs: Mapping[str, Path],
) -> None:
    set_seed(seed)
    model, tokenizer = load_lora_model(runtime, protocol, model_revision, DEVICE)
    try:
        base_generation = dirs["generations"] / f"{runtime.short_name}_base.jsonl"
        generate_behavior_sample(
            model,
            tokenizer,
            partitions.behavior_test,
            runtime.generation_microbatch,
            protocol.data_seed,
            base_generation,
            {
                "model": runtime.short_name,
                "seed": -1,
                "regime": "none",
                "condition": "base",
                "phase": -1,
                "checkpoint_kind": "base",
            },
            DEVICE,
        )

        aligned_state, _ = train_or_load_alignment(
            model,
            tokenizer,
            prepared,
            protocol,
            runtime,
            seed,
            dirs,
        )
        aligned_generation = dirs["generations"] / f"{runtime.short_name}_seed{seed}_aligned.jsonl"
        generate_behavior_sample(
            model,
            tokenizer,
            partitions.behavior_test,
            runtime.generation_microbatch,
            seed,
            aligned_generation,
            {
                "model": runtime.short_name,
                "seed": seed,
                "regime": "none",
                "condition": "aligned",
                "phase": 0,
                "checkpoint_kind": "aligned",
            },
            DEVICE,
        )

        for regime in protocol.regimes:
            for condition in protocol.conditions:
                restore_lora_state(model, aligned_state, DEVICE)
                run_branch(
                    model,
                    tokenizer,
                    prepared,
                    partitions,
                    protocol,
                    runtime,
                    seed,
                    regime,
                    condition,
                    aligned_state,
                    dirs,
                )
                release_cuda()
    finally:
        del model, tokenizer
        release_cuda()


def consolidate(dirs: Mapping[str, Path]) -> None:
    rows = []
    for path in sorted(dirs["runs"].glob("**/metrics.json")):
        rows.append(json.loads(path.read_text()))
    if rows:
        dataframe = pd.DataFrame(rows)
        keys = ["model", "seed", "regime", "condition", "phase"]
        dataframe = dataframe.drop_duplicates(keys, keep="last").sort_values(keys)
        dataframe.to_csv(dirs["results"] / "all_metrics.csv", index=False)
        LOG.info("Consolidated %d continual-learning rows", len(dataframe))

    alignment_rows = []
    for path in sorted(dirs["aligned"].glob("**/metrics.json")):
        raw = json.loads(path.read_text())
        row: Dict[str, Any] = {
            "model": raw["model"],
            "seed": raw["seed"],
            "alignment_train_loss": raw["alignment_train_loss"],
            "alignment_train_pairs": raw["alignment_train_pairs"],
            "alignment_train_target_tokens": raw["alignment_train_target_tokens"],
        }
        for stage in ("base_alignment", "post_alignment"):
            for key, value in raw[stage].items():
                row[f"{stage}_{key}"] = value
        alignment_rows.append(row)
    if alignment_rows:
        alignment = pd.DataFrame(alignment_rows).drop_duplicates(
            ["model", "seed"], keep="last"
        ).sort_values(["model", "seed"])
        alignment.to_csv(dirs["results"] / "alignment_stage_metrics.csv", index=False)
        LOG.info("Consolidated %d alignment-stage rows", len(alignment))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--assets", type=Path, default=Path("asset_manifest.json"))
    parser.add_argument("--root", type=Path, default=Path("pci_h100_outputs"))
    parser.add_argument("--models", nargs="+", choices=["8b", "14b"], default=["8b", "14b"])
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config: Config = load_config(arguments.config)
    assets = load_asset_manifest(arguments.assets)
    dirs = ensure_dirs(arguments.root)
    partitions = load_partitions(
        config.protocol,
        assets,
        dirs["manifests"] / "partition_manifest.json",
    )
    requested_seed_override = tuple(arguments.seeds) if arguments.seeds else None

    study_manifest = {
        "config": json.loads(arguments.config.read_text()),
        "assets": assets,
        "derived_alignment_partition_pairs": config.protocol.alignment_partition_pairs,
        "minimum_exact_test_seeds": config.protocol.minimum_exact_test_seeds,
        "model_seed_design": {
            key: {
                "seeds": list(model.seeds),
                "inference_role": model.inference_role,
            }
            for key, model in config.models.items()
        },
        "partition_counts": {
            "align_train": len(partitions.align_train),
            "align_control": len(partitions.align_control),
            "conflict_train": len(partitions.conflict_train),
            "conflict_eval": len(partitions.conflict_eval),
            "official_test": len(partitions.test),
            "behavior_test": len(partitions.behavior_test),
        },
        "partition_manifest": partitions.manifest,
    }
    study_path = dirs["root"] / "STUDY_MANIFEST.json"
    if study_path.exists():
        existing = json.loads(study_path.read_text())
        if existing != study_manifest:
            raise RuntimeError(
                "The existing output root was created with a different protocol, asset revision, "
                "or partition. Use a new --root rather than mixing experiments."
            )
    else:
        atomic_json(study_manifest, study_path)

    invocation_manifest = {
        "started_unix": time.time(),
        "environment": environment_manifest(),
        "models_requested": arguments.models,
        "seed_override": list(requested_seed_override) if requested_seed_override else None,
    }
    invocation_name = (
        f"invocation_{'_'.join(arguments.models)}_{time.time_ns()}.json"
    )
    atomic_json(invocation_manifest, dirs["manifests"] / invocation_name)

    for model_key in arguments.models:
        runtime = config.models[model_key]
        seeds = requested_seed_override if requested_seed_override is not None else runtime.seeds
        unknown = set(seeds) - set(runtime.seeds)
        if unknown:
            raise ValueError(
                f"Unregistered seeds for {model_key}: {sorted(unknown)}; "
                f"allowed={list(runtime.seeds)}"
            )
        revision = assets["models"][runtime.model_id]
        _, prepared = prepare_model_data(
            runtime,
            config.protocol,
            partitions,
            revision,
        )
        atomic_json(
            prepared.tokenizer_summary,
            dirs["manifests"] / f"tokenization_{runtime.short_name}.json",
        )
        for seed in seeds:
            run_model_seed(
                runtime,
                config.protocol,
                prepared,
                partitions,
                revision,
                seed,
                dirs,
            )
            consolidate(dirs)

    consolidate(dirs)
    LOG.info("TRAINING AND GENERATION COMPLETE")
    LOG.info("Next command: bash run_judge.sh")


if __name__ == "__main__":
    main()
