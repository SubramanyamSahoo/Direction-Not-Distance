#!/usr/bin/env python3
"""Independent evaluator with explicit validity checks.

The judge scores actual model generations. Before those scores are used, it is
calibrated on the same held-out HH chosen/rejected pairs. Constant outputs,
non-finite scores, or a mismatched evaluation set cause a hard failure rather
than a silent metric collapse.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from aaai27_core import DTYPE, atomic_json, load_asset_manifest, load_config, require_h100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger("aaai27.judge")
DEVICE = require_h100()


def install_armorm_transformers_compat() -> None:
    """Restore a docstring-only Llama symbol removed in newer Transformers.

    The pinned ArmoRM custom model imports LLAMA_INPUTS_DOCSTRING only to decorate
    its forward method. Transformers 4.56.x removed that constant. Reintroducing
    an empty string before trust_remote_code imports the custom module preserves
    model computation while restoring import compatibility.
    """
    import transformers
    from transformers.models.llama import modeling_llama

    if not hasattr(modeling_llama, "LLAMA_INPUTS_DOCSTRING"):
        modeling_llama.LLAMA_INPUTS_DOCSTRING = ""
        LOG.warning(
            "Installed ArmoRM compatibility shim for Transformers %s: "
            "LLAMA_INPUTS_DOCSTRING restored as an empty docstring.",
            transformers.__version__,
        )


def wilson_interval(successes: int, total: int, confidence: float) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p_hat = successes / total
    denominator = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denominator
    half = (
        z
        * ((p_hat * (1.0 - p_hat) / total + z * z / (4.0 * total * total)) ** 0.5)
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def conversation(messages: Sequence[Dict[str, str]], assistant_text: str) -> List[Dict[str, str]]:
    return list(messages) + [{"role": "assistant", "content": assistant_text}]


def format_conversations(tokenizer, conversations: Sequence[Sequence[Dict[str, str]]]) -> List[str]:
    return [
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
        for messages in conversations
    ]


def score_conversations(
    model,
    tokenizer,
    conversations: Sequence[Sequence[Dict[str, str]]],
    batch_size: int,
    max_length: int,
) -> Tuple[List[float], List[float]]:
    """Length-bucketed scoring with deterministic CUDA OOM backoff."""
    texts = format_conversations(tokenizer, conversations)
    lengths = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in texts
    ]
    order = sorted(range(len(texts)), key=lambda index: (lengths[index], index))
    scores: List[float] = [float("nan")] * len(texts)
    at_limit: List[float] = [0.0] * len(texts)
    position = 0
    active_batch_size = max(1, int(batch_size))

    while position < len(order):
        take = min(active_batch_size, len(order) - position)
        indices = order[position : position + take]
        chunk = [texts[index] for index in indices]
        try:
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(DEVICE)
            batch_at_limit = (
                encoded["attention_mask"]
                .sum(dim=1)
                .eq(max_length)
                .float()
                .cpu()
                .tolist()
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
                output = model(**encoded)
            if not hasattr(output, "score"):
                raise RuntimeError("ArmoRM output has no .score field")
            tensor = output.score.detach().float().reshape(-1)
            if tensor.numel() != len(chunk):
                raise RuntimeError(
                    f"Judge returned {tensor.numel()} scores for batch size {len(chunk)}"
                )
            values = tensor.cpu().tolist()
            for index, value, limit_flag in zip(indices, values, batch_at_limit):
                scores[index] = float(value)
                at_limit[index] = float(limit_flag)
            position += take
            # Return gradually toward the requested batch size after a successful retry.
            active_batch_size = min(int(batch_size), max(active_batch_size, take) * 2)
            del encoded, output, tensor
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if take == 1:
                raise RuntimeError(
                    f"Judge OOM on a single sequence of {lengths[indices[0]]} tokens"
                )
            active_batch_size = max(1, take // 2)
            LOG.warning(
                "Judge CUDA OOM; retrying the same length bucket with batch size %d",
                active_batch_size,
            )

    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise RuntimeError("Judge scoring left non-finite or unfilled outputs")
    return scores, at_limit


def validate_generation_sets(files: Sequence[Path]) -> List[Dict[str, Any]]:
    first = read_jsonl(files[0])
    if not first:
        raise RuntimeError(f"Empty generation file: {files[0]}")
    canonical_ids = {row["id"] for row in first}
    if len(canonical_ids) != len(first):
        raise RuntimeError("Duplicate behavior IDs in generation file")
    canonical_content = {
        row["id"]: (row["messages"], row["chosen"], row["rejected"])
        for row in first
    }
    for path in files[1:]:
        rows = read_jsonl(path)
        ids = {row["id"] for row in rows}
        if len(ids) != len(rows):
            raise RuntimeError(f"Duplicate behavior IDs in {path}")
        if ids != canonical_ids:
            raise RuntimeError(
                f"Generation set mismatch in {path}; every checkpoint must use the same IDs"
            )
        for row in rows:
            if (row["messages"], row["chosen"], row["rejected"]) != canonical_content[row["id"]]:
                raise RuntimeError(f"Reference content mismatch for {row['id']} in {path}")
    return sorted(first, key=lambda row: row["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("pci_h100_outputs"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--assets", type=Path, default=Path("asset_manifest.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    assets = load_asset_manifest(arguments.assets)
    generation_files = sorted((arguments.root / "generations").glob("*.jsonl"))
    if not generation_files:
        raise FileNotFoundError(f"No generation files in {arguments.root / 'generations'}")
    reference_rows = validate_generation_sets(generation_files)

    judge_id = config.protocol.judge_model
    revision = assets["models"][judge_id]
    LOG.info("Loading judge %s@%s", judge_id, revision)
    install_armorm_transformers_compat()
    tokenizer = AutoTokenizer.from_pretrained(
        judge_id,
        revision=revision,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        judge_id,
        revision=revision,
        trust_remote_code=True,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    context_candidates = [
        int(value)
        for value in (
            getattr(model.config, "max_position_embeddings", 0),
            getattr(tokenizer, "model_max_length", 0),
        )
        if isinstance(value, int) and 0 < value < 1_000_000
    ]
    if not context_candidates:
        raise RuntimeError("Could not derive the judge context limit from model/tokenizer metadata")
    judge_context_limit = min(context_candidates)
    LOG.info("Judge context limit derived from metadata: %d", judge_context_limit)

    results_dir = arguments.root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    reference_path = results_dir / "judge_reference_scores.csv"
    validity_path = results_dir / "judge_validity.json"

    if reference_path.exists() and validity_path.exists():
        references = pd.read_csv(reference_path)
        expected_ids = {row["id"] for row in reference_rows}
        if set(references["id"].astype(str)) != {str(x) for x in expected_ids}:
            raise RuntimeError(
                "Cached judge reference IDs do not match the current behavior sample; "
                "remove judge_reference_scores.csv and rerun the judge."
            )
        validity = json.loads(validity_path.read_text())
        if not bool(validity.get("calibration_passed", False)):
            raise RuntimeError(
                "Cached judge calibration did not pass the preregistered validity gate; "
                "remove cached judge files only after correcting the evaluator."
            )
        LOG.info("Using cached reference calibration")
    else:
        conversations: List[List[Dict[str, str]]] = []
        for row in reference_rows:
            conversations.append(conversation(row["messages"], row["chosen"]))
            conversations.append(conversation(row["messages"], row["rejected"]))
        values, reference_at_limit = score_conversations(
            model,
            tokenizer,
            conversations,
            arguments.batch_size,
            judge_context_limit,
        )
        chosen = np.asarray(values[0::2], dtype=np.float64)
        rejected = np.asarray(values[1::2], dtype=np.float64)
        chosen_at_limit = np.asarray(reference_at_limit[0::2], dtype=np.float64)
        rejected_at_limit = np.asarray(reference_at_limit[1::2], dtype=np.float64)
        if not np.isfinite(chosen).all() or not np.isfinite(rejected).all():
            raise RuntimeError("Judge produced non-finite reference scores")
        combined = np.concatenate([chosen, rejected])
        if np.ptp(combined) == 0:
            raise RuntimeError("Judge produced a constant score and is invalid for this evaluation")
        references = pd.DataFrame(
            {
                "id": [row["id"] for row in reference_rows],
                "judge_chosen": chosen,
                "judge_rejected": rejected,
                "judge_chosen_input_at_limit": chosen_at_limit,
                "judge_rejected_input_at_limit": rejected_at_limit,
            }
        )
        references["judge_reference_correct"] = (
            references["judge_chosen"] > references["judge_rejected"]
        ).astype(float)
        references["judge_reference_tie"] = (
            references["judge_chosen"] == references["judge_rejected"]
        ).astype(float)
        correct_count = int(references["judge_reference_correct"].sum())
        lower, upper = wilson_interval(
            correct_count,
            len(references),
            config.protocol.behavior_confidence,
        )
        calibration_passed = bool(lower > 0.5)
        references.to_csv(reference_path, index=False)
        validity = {
            "judge_model": judge_id,
            "judge_revision": revision,
            "examples": int(len(references)),
            "chosen_over_rejected_accuracy": float(references["judge_reference_correct"].mean()),
            "chosen_over_rejected_wilson_lower": lower,
            "chosen_over_rejected_wilson_upper": upper,
            "validity_confidence": config.protocol.behavior_confidence,
            "validity_gate": "Wilson lower bound strictly greater than 0.5",
            "calibration_passed": calibration_passed,
            "tie_fraction": float(references["judge_reference_tie"].mean()),
            "chosen_score_mean": float(references["judge_chosen"].mean()),
            "rejected_score_mean": float(references["judge_rejected"].mean()),
            "combined_score_std": float(combined.std(ddof=1)),
            "chosen_input_at_max_length_fraction": float(chosen_at_limit.mean()),
            "rejected_input_at_max_length_fraction": float(rejected_at_limit.mean()),
            "judge_max_length": judge_context_limit,
            "constant_output": False,
        }
        atomic_json(validity, validity_path)
        if not calibration_passed:
            raise RuntimeError(
                "Judge failed calibration: the lower confidence bound for chosen-over-rejected "
                f"accuracy is {lower:.4f}, not strictly above chance. Validity={validity}"
            )

    all_scored: List[pd.DataFrame] = []
    for path in generation_files:
        output_path = results_dir / f"judge_{path.stem}.csv"
        if output_path.exists():
            LOG.info("SKIP scored %s", path.name)
            all_scored.append(pd.read_csv(output_path))
            continue
        rows = read_jsonl(path)
        generated_conversations = [
            conversation(row["messages"], row["generated"])
            for row in rows
        ]
        generated_scores, generated_at_limit = score_conversations(
            model,
            tokenizer,
            generated_conversations,
            arguments.batch_size,
            judge_context_limit,
        )
        generated_array = np.asarray(generated_scores, dtype=np.float64)
        if not np.isfinite(generated_array).all():
            raise RuntimeError(f"Judge produced non-finite generated scores for {path}")
        dataframe = pd.DataFrame(
            {
                "id": [row["id"] for row in rows],
                "model": [row["model"] for row in rows],
                "seed": [row["seed"] for row in rows],
                "regime": [row["regime"] for row in rows],
                "condition": [row["condition"] for row in rows],
                "phase": [row["phase"] for row in rows],
                "checkpoint_kind": [row["checkpoint_kind"] for row in rows],
                "generated_length_chars": [len(row["generated"]) for row in rows],
                "generated_empty": [float(not row["generated"].strip()) for row in rows],
                "judge_input_at_limit": generated_at_limit,
                "judge_generated": generated_scores,
            }
        )
        dataframe = dataframe.merge(references, on="id", how="left", validate="one_to_one")
        if dataframe[["judge_chosen", "judge_rejected"]].isna().any().any():
            raise RuntimeError("Reference score merge failed")
        dataframe["judge_win_vs_rejected"] = (
            dataframe["judge_generated"] > dataframe["judge_rejected"]
        ).astype(float)
        dataframe["judge_gap_vs_rejected"] = (
            dataframe["judge_generated"] - dataframe["judge_rejected"]
        )
        dataframe["judge_gap_to_chosen"] = (
            dataframe["judge_generated"] - dataframe["judge_chosen"]
        )
        dataframe.to_csv(output_path, index=False)
        all_scored.append(dataframe)
        LOG.info("Scored %s (%d generations)", path.name, len(dataframe))

    raw = pd.concat(all_scored, ignore_index=True)
    raw.to_csv(results_dir / "behavior_judge_raw.csv", index=False)
    keys = ["model", "seed", "regime", "condition", "phase", "checkpoint_kind"]
    aggregate = raw.groupby(keys, dropna=False).agg(
        judge_generated_mean=("judge_generated", "mean"),
        judge_generated_std=("judge_generated", "std"),
        judge_win_vs_rejected=("judge_win_vs_rejected", "mean"),
        judge_gap_vs_rejected=("judge_gap_vs_rejected", "mean"),
        judge_gap_to_chosen=("judge_gap_to_chosen", "mean"),
        generated_length_chars_mean=("generated_length_chars", "mean"),
        generated_empty_fraction=("generated_empty", "mean"),
        judge_input_at_max_length_fraction=("judge_input_at_limit", "mean"),
        judge_examples=("id", "count"),
    ).reset_index()
    aggregate.to_csv(results_dir / "behavior_judge_aggregate.csv", index=False)
    LOG.info("JUDGE COMPLETE; validity=%s", validity)


if __name__ == "__main__":
    main()
