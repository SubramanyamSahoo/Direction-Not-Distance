#!/usr/bin/env python3
"""Shared implementation for the AAAI-27 directional-update experiment."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG = logging.getLogger("aaai27")
DTYPE = torch.bfloat16
TURN_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*")


@dataclass(frozen=True)
class ModelRuntime:
    model_id: str
    short_name: str
    pair_microbatch: int
    pair_grad_accum: int
    shift_microbatch: int
    shift_grad_accum: int
    eval_microbatch: int
    control_microbatch: int
    generation_microbatch: int
    seeds: Tuple[int, ...]
    inference_role: str


@dataclass(frozen=True)
class Protocol:
    max_length: int
    tokens_per_phase: int
    num_phases: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_clip: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: Tuple[str, ...]
    data_seed: int
    hh_dataset: str
    hh_config: str
    owt_dataset: str
    judge_model: str
    regimes: Tuple[str, ...]
    conditions: Tuple[str, ...]
    behavior_confidence: float
    behavior_half_width: float

    @property
    def alignment_partition_pairs(self) -> int:
        """Derived from the prior protocol's token-normalized phase budget."""
        if self.tokens_per_phase % self.max_length != 0:
            raise ValueError("tokens_per_phase must be divisible by max_length")
        return self.tokens_per_phase // self.max_length

    @property
    def minimum_exact_test_seeds(self) -> int:
        """Smallest n allowing Holm-adjusted p<.05 for all pairwise methods."""
        comparisons = math.comb(len(self.conditions), 2)
        n = 1
        # The smallest attainable two-sided sign-flip p-value is 2 / 2^n.
        # Holm's most stringent multiplier is the family size.
        while comparisons * 2 / (2 ** n) >= 0.05:
            n += 1
        return n


@dataclass(frozen=True)
class Config:
    protocol: Protocol
    models: Dict[str, ModelRuntime]


@dataclass
class DataPartitions:
    align_train: List[Dict[str, Any]]
    align_control: List[Dict[str, Any]]
    conflict_train: List[Dict[str, Any]]
    conflict_eval: List[Dict[str, Any]]
    test: List[Dict[str, Any]]
    behavior_test: List[Dict[str, Any]]
    owt_train: List[str]
    owt_eval: List[str]
    manifest: Dict[str, Any]


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text())
    p = raw["protocol"]
    protocol = Protocol(
        max_length=int(p["max_length"]),
        tokens_per_phase=int(p["tokens_per_phase"]),
        num_phases=int(p["num_phases"]),
        learning_rate=float(p["learning_rate"]),
        weight_decay=float(p["weight_decay"]),
        warmup_ratio=float(p["warmup_ratio"]),
        grad_clip=float(p["grad_clip"]),
        lora_r=int(p["lora_r"]),
        lora_alpha=int(p["lora_alpha"]),
        lora_dropout=float(p["lora_dropout"]),
        lora_targets=tuple(p["lora_targets"]),
        data_seed=int(p["data_seed"]),
        hh_dataset=str(p["hh_dataset"]),
        hh_config=str(p["hh_config"]),
        owt_dataset=str(p["owt_dataset"]),
        judge_model=str(p["judge_model"]),
        regimes=tuple(p["regimes"]),
        conditions=tuple(p["conditions"]),
        behavior_confidence=float(p["behavior_confidence"]),
        behavior_half_width=float(p["behavior_half_width"]),
    )
    supported_conditions = {"unconstrained", "global_projection", "coordinate_mortality"}
    if set(protocol.conditions) != supported_conditions:
        raise ValueError(f"Conditions must be exactly {sorted(supported_conditions)}")
    supported_regimes = {"benign", "conflict"}
    if set(protocol.regimes) != supported_regimes:
        raise ValueError(f"Regimes must be exactly {sorted(supported_regimes)}")

    models: Dict[str, ModelRuntime] = {}
    for key, value in raw["models"].items():
        models[key] = ModelRuntime(
            model_id=str(value["model_id"]),
            short_name=str(value["short_name"]),
            pair_microbatch=int(value["pair_microbatch"]),
            pair_grad_accum=int(value["pair_grad_accum"]),
            shift_microbatch=int(value["shift_microbatch"]),
            shift_grad_accum=int(value["shift_grad_accum"]),
            eval_microbatch=int(value["eval_microbatch"]),
            control_microbatch=int(value["control_microbatch"]),
            generation_microbatch=int(value["generation_microbatch"]),
            seeds=tuple(int(x) for x in value["seeds"]),
            inference_role=str(value["inference_role"]),
        )
        if len(models[key].seeds) < 3:
            raise ValueError(f"Model {key} must have at least three paired seeds")
        if models[key].inference_role == "primary" and len(models[key].seeds) < protocol.minimum_exact_test_seeds:
            raise ValueError(
                f"Primary model {key} needs at least {protocol.minimum_exact_test_seeds} seeds "
                "for the preregistered Holm-adjusted exact-test resolution"
            )
        if models[key].inference_role not in {"primary", "scale_replication"}:
            raise ValueError(f"Unknown inference_role for {key}: {models[key].inference_role}")
    if sum(model.inference_role == "primary" for model in models.values()) != 1:
        raise ValueError("Exactly one model must be designated as the primary inferential scale")
    return Config(protocol=protocol, models=models)


def load_asset_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python prepare_assets.py before offline training."
        )
    obj = json.loads(path.read_text())
    if "models" not in obj or "datasets" not in obj:
        raise ValueError("Malformed asset manifest")
    return obj


def require_h100() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "H100" not in name.upper():
        raise RuntimeError(f"Expected NVIDIA H100; detected {name}")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory / (1024 ** 3) < 75:
        raise RuntimeError("Expected an 80 GB H100 class GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    LOG.info(
        "GPU=%s VRAM=%.1f GiB torch=%s torch_cuda=%s",
        name,
        props.total_memory / (1024 ** 3),
        torch.__version__,
        torch.version.cuda,
    )
    return torch.device("cuda:0")


def environment_manifest() -> Dict[str, Any]:
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,pci.bus_id", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        smi = "unavailable"
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
    }


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "aligned": root / "aligned",
        "runs": root / "runs",
        "generations": root / "generations",
        "results": root / "results",
        "logs": root / "logs",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def append_jsonl(row: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def parse_hh_transcript(text: str) -> List[Dict[str, str]]:
    matches = list(TURN_RE.finditer(text))
    if not matches:
        raise ValueError("No HH turns found")
    messages: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        role = "user" if match.group(1) == "Human" else "assistant"
        content = text[start:end].strip()
        if not content:
            raise ValueError("Empty HH turn")
        messages.append({"role": role, "content": content})
    return messages


def hh_pair_to_record(chosen: str, rejected: str, source_id: str) -> Dict[str, Any]:
    c = parse_hh_transcript(chosen)
    r = parse_hh_transcript(rejected)
    if c[-1]["role"] != "assistant" or r[-1]["role"] != "assistant":
        raise ValueError("Pair does not end in assistant turns")
    if c[:-1] != r[:-1]:
        raise ValueError("Chosen and rejected histories differ")
    return {
        "id": source_id,
        "messages": c[:-1],
        "chosen": c[-1]["content"],
        "rejected": r[-1]["content"],
    }


def wilson_required_n(confidence: float, half_width: float) -> int:
    """Worst-case Bernoulli sample size for a Wilson interval precision target."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    if not 0 < half_width < 0.5:
        raise ValueError("half_width must lie in (0,0.5)")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    n = 1
    while True:
        p = 0.5
        denom = 1 + z * z / n
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        if half <= half_width:
            return n
        n += 1


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derived_seed(*parts: Any) -> int:
    """Derive a reproducible 31-bit seed without hand-chosen arithmetic constants."""
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def load_partitions(
    protocol: Protocol,
    assets: Mapping[str, Mapping[str, str]],
    output_manifest: Optional[Path] = None,
) -> DataPartitions:
    hh_revision = assets["datasets"][protocol.hh_dataset]
    owt_revision = assets["datasets"][protocol.owt_dataset]
    # Resolve the exact cached Hub snapshot first, then load the raw local files.
    # This avoids Datasets' generated cache-config names, which are not stable
    # across online/offline resolution paths. The training run therefore remains
    # fully offline after prepare_assets.py has cached the repository snapshots.
    LOG.info("Loading cached %s@%s / %s", protocol.hh_dataset, hh_revision, protocol.hh_config)
    hh_root = Path(
        snapshot_download(
            repo_id=protocol.hh_dataset,
            repo_type="dataset",
            revision=hh_revision,
            local_files_only=True,
        )
    )
    hh_train_file = hh_root / protocol.hh_config / "train.jsonl.gz"
    hh_test_file = hh_root / protocol.hh_config / "test.jsonl.gz"
    for required_file in (hh_train_file, hh_test_file):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required cached HH-RLHF file is missing: {required_file}")
    hh_train = load_dataset(
        "json",
        data_files={"train": str(hh_train_file)},
        split="train",
    )
    hh_test = load_dataset(
        "json",
        data_files={"test": str(hh_test_file)},
        split="test",
    )
    hh_train = hh_train.add_column("_source_index", list(range(len(hh_train))))
    hh_test = hh_test.add_column("_source_index", list(range(len(hh_test))))
    hh_train = hh_train.shuffle(seed=protocol.data_seed)

    valid_train: List[Dict[str, Any]] = []
    invalid_train = 0
    for ex in hh_train:
        try:
            valid_train.append(
                hh_pair_to_record(
                    ex["chosen"],
                    ex["rejected"],
                    f"train:{int(ex['_source_index'])}",
                )
            )
        except ValueError:
            invalid_train += 1

    valid_test: List[Dict[str, Any]] = []
    invalid_test = 0
    for ex in hh_test:
        try:
            valid_test.append(
                hh_pair_to_record(
                    ex["chosen"],
                    ex["rejected"],
                    f"test:{int(ex['_source_index'])}",
                )
            )
        except ValueError:
            invalid_test += 1

    n = protocol.alignment_partition_pairs
    conflict_eval_count = len(valid_test)
    required = 2 * n + conflict_eval_count + 1
    if len(valid_train) < required:
        raise RuntimeError(
            "Not enough valid HH examples for disjoint alignment, control, "
            "conflict-train, and conflict-evaluation partitions"
        )
    align_train = valid_train[:n]
    align_control = valid_train[n : 2 * n]
    conflict_eval = valid_train[2 * n : 2 * n + conflict_eval_count]
    conflict_train = valid_train[2 * n + conflict_eval_count :]
    test = valid_test

    behavior_n = min(
        len(test),
        wilson_required_n(protocol.behavior_confidence, protocol.behavior_half_width),
    )
    behavior_test = sorted(
        test,
        key=lambda row: stable_hash(f"{protocol.data_seed}:{row['id']}"),
    )[:behavior_n]

    LOG.info("Loading cached %s@%s", protocol.owt_dataset, owt_revision)
    owt_root = Path(
        snapshot_download(
            repo_id=protocol.owt_dataset,
            repo_type="dataset",
            revision=owt_revision,
            local_files_only=True,
        )
    )
    parquet_files = sorted((owt_root / "data").glob("*.parquet"))
    if not parquet_files:
        parquet_files = sorted(owt_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No cached Parquet data files found under {owt_root} for {protocol.owt_dataset}"
        )
    owt = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in parquet_files]},
        split="train",
    )
    owt_original_count = len(owt)
    owt = owt.add_column("_source_index", list(range(len(owt))))
    owt = owt.filter(
        lambda example: isinstance(example.get("text"), str)
        and bool(example["text"].strip()),
        load_from_cache_file=True,
    )
    owt = owt.shuffle(seed=protocol.data_seed)
    if len(owt) <= len(test):
        raise RuntimeError("OpenWebText subset is too small for matched held-out evaluation")
    owt_eval_slice = owt.select(range(len(test)))
    owt_train_slice = owt.select(range(len(test), len(owt)))
    owt_eval = [str(x) for x in owt_eval_slice["text"]]
    owt_train = [str(x) for x in owt_train_slice["text"]]
    owt_eval_indices = [int(x) for x in owt_eval_slice["_source_index"]]
    owt_train_indices = [int(x) for x in owt_train_slice["_source_index"]]

    manifest = {
        "hh_revision": hh_revision,
        "owt_revision": owt_revision,
        "invalid_train_pairs": invalid_train,
        "invalid_test_pairs": invalid_test,
        "alignment_train_count": len(align_train),
        "alignment_control_count": len(align_control),
        "conflict_train_count": len(conflict_train),
        "conflict_eval_count": len(conflict_eval),
        "official_test_count": len(test),
        "behavior_sample_count": len(behavior_test),
        "behavior_confidence": protocol.behavior_confidence,
        "behavior_half_width": protocol.behavior_half_width,
        "alignment_train_ids": [x["id"] for x in align_train],
        "alignment_control_ids": [x["id"] for x in align_control],
        "behavior_test_ids": [x["id"] for x in behavior_test],
        "official_test_id_sha256": stable_hash("\n".join(x["id"] for x in test)),
        "conflict_train_id_sha256": stable_hash("\n".join(x["id"] for x in conflict_train)),
        "conflict_eval_id_sha256": stable_hash("\n".join(x["id"] for x in conflict_eval)),
        "owt_original_count": owt_original_count,
        "owt_nonempty_count": len(owt),
        "owt_empty_or_whitespace_removed": owt_original_count - len(owt),
        "owt_eval_source_index_sha256": stable_hash("\n".join(map(str, owt_eval_indices))),
        "owt_train_source_index_sha256": stable_hash("\n".join(map(str, owt_train_indices))),
    }
    if output_manifest is not None:
        atomic_json(manifest, output_manifest)

    LOG.info(
        "Partitions align=%d control=%d conflict_train=%d conflict_eval=%d test=%d behavior=%d OWTtrain=%d OWTeval=%d",
        len(align_train),
        len(align_control),
        len(conflict_train),
        len(conflict_eval),
        len(test),
        len(behavior_test),
        len(owt_train),
        len(owt_eval),
    )
    return DataPartitions(
        align_train=align_train,
        align_control=align_control,
        conflict_train=conflict_train,
        conflict_eval=conflict_eval,
        test=test,
        behavior_test=behavior_test,
        owt_train=owt_train,
        owt_eval=owt_eval,
        manifest=manifest,
    )


def get_tokenizer(model_id: str, revision: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def qwen_prompt_text(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _response_ids(tokenizer, response: str) -> List[int]:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    return ids


def encode_pair_shared_prompt(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    chosen: str,
    rejected: str,
    max_length: int,
) -> Dict[str, Any]:
    """Encode both responses with exactly the same truncated prompt suffix."""
    prompt_ids = tokenizer(
        qwen_prompt_text(tokenizer, messages),
        add_special_tokens=False,
    )["input_ids"]
    chosen_ids = _response_ids(tokenizer, chosen)
    rejected_ids = _response_ids(tokenizer, rejected)
    response_truncated = max(len(chosen_ids), len(rejected_ids)) >= max_length
    if response_truncated:
        shared_prompt: List[int] = []
        chosen_ids = chosen_ids[:max_length]
        rejected_ids = rejected_ids[:max_length]
    else:
        keep_prompt = max_length - max(len(chosen_ids), len(rejected_ids))
        shared_prompt = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
    prompt_dropped = max(len(prompt_ids) - len(shared_prompt), 0)

    c = shared_prompt + chosen_ids
    r = shared_prompt + rejected_ids
    return {
        "chosen_ids": c,
        "chosen_mask": [0] * len(shared_prompt) + [1] * len(chosen_ids),
        "rejected_ids": r,
        "rejected_mask": [0] * len(shared_prompt) + [1] * len(rejected_ids),
        "response_truncated": response_truncated,
        "prompt_tokens_dropped": prompt_dropped,
    }


def encode_single_response(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    response: str,
    max_length: int,
) -> Dict[str, List[int]]:
    prompt_ids = tokenizer(
        qwen_prompt_text(tokenizer, messages),
        add_special_tokens=False,
    )["input_ids"]
    response_ids = _response_ids(tokenizer, response)
    if len(response_ids) >= max_length:
        prompt_ids = []
        response_ids = response_ids[:max_length]
    else:
        prompt_ids = prompt_ids[-(max_length - len(response_ids)) :]
    ids = prompt_ids + response_ids
    return {
        "input_ids": ids,
        "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
    }


class PairDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer, max_length: int):
        self.items: List[Dict[str, Any]] = []
        for row in tqdm(records, desc="tokenize preference pairs", leave=False):
            encoded = encode_pair_shared_prompt(
                tokenizer,
                row["messages"],
                row["chosen"],
                row["rejected"],
                max_length,
            )
            self.items.append({"id": row["id"], **encoded})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def truncation_summary(self) -> Dict[str, float]:
        if not self.items:
            return {"response_truncation_fraction": float("nan"), "mean_prompt_tokens_dropped": float("nan")}
        return {
            "response_truncation_fraction": float(np.mean([x["response_truncated"] for x in self.items])),
            "mean_prompt_tokens_dropped": float(np.mean([x["prompt_tokens_dropped"] for x in self.items])),
        }


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


class ConflictDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def pad_sequences(sequences: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(x) for x in sequences)
    ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention[index, :length] = 1
    return ids, attention


class PairCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen = [x["chosen_ids"] for x in batch]
        rejected = [x["rejected_ids"] for x in batch]
        masks = [x["chosen_mask"] for x in batch] + [x["rejected_mask"] for x in batch]
        ids, attention = pad_sequences(chosen + rejected, self.pad_id)
        response_mask = torch.zeros_like(ids)
        for index, mask in enumerate(masks):
            response_mask[index, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attention,
            "response_mask": response_mask,
            "pair_batch_size": torch.tensor(len(batch), dtype=torch.long),
        }


class TextCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[str]) -> Dict[str, torch.Tensor]:
        eos = self.tokenizer.eos_token or ""
        texts = [text.rstrip() + eos for text in batch]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=False,
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        if int(labels[:, 1:].ne(-100).sum().item()) == 0:
            raise RuntimeError("Benign-shift batch has zero causal-LM target tokens")
        encoded["labels"] = labels
        return encoded


class ConflictCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded = [
            encode_single_response(self.tokenizer, x["messages"], x["rejected"], self.max_length)
            for x in batch
        ]
        ids, attention = pad_sequences([x["input_ids"] for x in encoded], self.tokenizer.pad_token_id)
        labels = ids.clone()
        for index, item in enumerate(encoded):
            mask = torch.tensor(item["response_mask"], dtype=torch.bool)
            labels[index, : len(mask)][~mask] = -100
        labels[attention == 0] = -100
        return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def make_pair_loader(
    dataset: PairDataset,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=PairCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def make_shift_loader(
    regime: str,
    partitions: DataPartitions,
    tokenizer,
    protocol: Protocol,
    runtime: ModelRuntime,
    seed: int,
    phase: int,
    eval_mode: bool,
) -> DataLoader:
    loader_seed = seed * 1009 + phase * 101 + (0 if regime == "benign" else 1)
    generator = torch.Generator().manual_seed(loader_seed)
    if regime == "benign":
        dataset: Dataset = TextDataset(partitions.owt_eval if eval_mode else partitions.owt_train)
        collator = TextCollator(tokenizer, protocol.max_length)
    elif regime == "conflict":
        dataset = ConflictDataset(
            partitions.conflict_eval if eval_mode else partitions.conflict_train
        )
        collator = ConflictCollator(tokenizer, protocol.max_length)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return DataLoader(
        dataset,
        batch_size=runtime.eval_microbatch if eval_mode else runtime.shift_microbatch,
        shuffle=not eval_mode,
        generator=generator if not eval_mode else None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def load_lora_model(
    runtime: ModelRuntime,
    protocol: Protocol,
    revision: str,
    device: torch.device,
):
    tokenizer = get_tokenizer(runtime.model_id, revision)
    LOG.info("Loading %s@%s in BF16", runtime.model_id, revision)
    base = AutoModelForCausalLM.from_pretrained(
        runtime.model_id,
        revision=revision,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    base.to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=protocol.lora_r,
        lora_alpha=protocol.lora_alpha,
        lora_dropout=protocol.lora_dropout,
        target_modules=list(protocol.lora_targets),
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    model.to(device)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Trainable LoRA parameters=%d (%.5f%% of %d)", trainable, 100 * trainable / total, total)
    return model, tokenizer


def lora_named_params(model) -> List[Tuple[str, nn.Parameter]]:
    params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    ]
    if not params:
        raise RuntimeError("No trainable LoRA parameters found")
    return params


def lora_state_cpu(model) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().contiguous() for name, parameter in lora_named_params(model)}


def save_lora_state(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.safetensors")
    safe_save(lora_state_cpu(model), str(tmp))
    os.replace(tmp, path)


def load_lora_state(model, path: Path) -> None:
    state = safe_load(str(path), device="cpu")
    named = dict(model.named_parameters())
    expected = {name for name, _ in lora_named_params(model)}
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise RuntimeError(f"LoRA state mismatch missing={missing[:3]} extra={extra[:3]}")
    with torch.no_grad():
        for name, tensor_value in state.items():
            named[name].copy_(tensor_value.to(named[name].device, dtype=named[name].dtype))


def restore_lora_state(model, state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor_value in state.items():
            named[name].copy_(tensor_value.to(device, dtype=named[name].dtype))


def lora_vector(model) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1).float() for _, parameter in lora_named_params(model)])


def state_vector(state: Mapping[str, torch.Tensor], model, device: torch.device) -> torch.Tensor:
    order = [name for name, _ in lora_named_params(model)]
    return torch.cat([state[name].to(device=device, dtype=torch.float32).reshape(-1) for name in order])


def write_lora_vector(model, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, parameter in lora_named_params(model):
            count = parameter.numel()
            parameter.copy_(vector[offset : offset + count].view_as(parameter).to(parameter.dtype))
            offset += count
    if offset != vector.numel():
        raise RuntimeError("LoRA vector writeback length mismatch")


def response_mean_logp(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probabilities = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]
    selected = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:].float()
    counts = mask.sum(dim=-1).clamp_min(1.0)
    return (selected * mask).sum(dim=-1) / counts, counts


def pairwise_ranking_loss(model, batch: Dict[str, torch.Tensor], device: torch.device):
    ids = batch["input_ids"].to(device, non_blocking=True)
    attention = batch["attention_mask"].to(device, non_blocking=True)
    response_mask = batch["response_mask"].to(device, non_blocking=True)
    pair_batch = int(batch["pair_batch_size"].item())
    with torch.autocast("cuda", dtype=DTYPE):
        output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    mean_logp, counts = response_mean_logp(output.logits, ids, response_mask)
    chosen = mean_logp[:pair_batch]
    rejected = mean_logp[pair_batch:]
    margins = chosen - rejected
    loss = F.softplus(-margins).mean()
    return loss, {
        "margins": margins,
        "chosen_logp": chosen,
        "rejected_logp": rejected,
        "target_tokens": counts.sum(),
    }


@torch.no_grad()
def evaluate_alignment(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    margins: List[torch.Tensor] = []
    chosen_values: List[torch.Tensor] = []
    rejected_values: List[torch.Tensor] = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        response_mask = batch["response_mask"].to(device, non_blocking=True)
        pair_batch = int(batch["pair_batch_size"].item())
        with torch.autocast("cuda", dtype=DTYPE):
            output = model(input_ids=ids, attention_mask=attention, use_cache=False)
        mean_logp, _ = response_mean_logp(output.logits, ids, response_mask)
        chosen = mean_logp[:pair_batch]
        rejected = mean_logp[pair_batch:]
        margins.append(chosen - rejected)
        chosen_values.append(chosen)
        rejected_values.append(rejected)
    margin = torch.cat(margins)
    chosen = torch.cat(chosen_values)
    rejected = torch.cat(rejected_values)
    model.train(was_training)
    return {
        "alignment_accuracy": (margin > 0).float().mean().item(),
        "alignment_margin_mean": margin.mean().item(),
        "alignment_margin_median": margin.median().item(),
        "chosen_mean_logp": chosen.mean().item(),
        "rejected_mean_logp": rejected.mean().item(),
        "alignment_pairs": int(margin.numel()),
    }


def full_alignment_gradient(model, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, float]:
    """Deterministic full-control-set gradient with dropout disabled."""
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    total_examples = len(loader.dataset)
    seen = 0
    weighted_loss = 0.0
    with torch.enable_grad():
        for batch in loader:
            batch_size = int(batch["pair_batch_size"].item())
            loss, _ = pairwise_ranking_loss(model, batch, device)
            (loss * (batch_size / total_examples)).backward()
            seen += batch_size
            weighted_loss += loss.detach().float().item() * batch_size
    if seen != total_examples:
        raise RuntimeError(f"Control gradient saw {seen}/{total_examples} pairs")
    gradients: List[torch.Tensor] = []
    for _, parameter in lora_named_params(model):
        gradient = parameter.grad
        gradients.append(
            torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
            if gradient is None
            else gradient.detach().float().reshape(-1).clone()
        )
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return torch.cat(gradients), weighted_loss / total_examples


@torch.no_grad()
def evaluate_shift_nll(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_sum = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=DTYPE):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        log_probabilities = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        safe_targets = targets.masked_fill(~mask, 0)
        selected = log_probabilities.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        nll_sum += -(selected * mask).sum().double()
        token_sum += mask.sum().double()
    mean_nll = (nll_sum / token_sum.clamp_min(1)).item()
    model.train(was_training)
    return {
        "shift_nll": mean_nll,
        "shift_perplexity": math.exp(min(mean_nll, 50.0)),
        "shift_eval_tokens": int(token_sum.item()),
    }


@torch.no_grad()
def drift_metrics(model, anchor_vector: torch.Tensor) -> Dict[str, float]:
    current = lora_vector(model)
    difference = current - anchor_vector
    denominator = anchor_vector.norm().clamp_min(torch.finfo(torch.float32).eps)
    return {
        "l2_drift": difference.norm().item(),
        "relative_l2_drift": (difference.norm() / denominator).item(),
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if denominator.item() == 0:
        return float("nan")
    return torch.dot(left, right).div(denominator).item()


def token_lr_multiplier(processed: int, target: int, warmup_ratio: float) -> float:
    progress = min(max(processed / max(target, 1), 0.0), 1.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        return progress / warmup_ratio
    remaining = (progress - warmup_ratio) / max(1 - warmup_ratio, 1e-12)
    remaining = min(max(remaining, 0.0), 1.0)
    return 0.5 * (1 + math.cos(math.pi * remaining))


def set_optimizer_lr(optimizer: AdamW, base_lr: float, multiplier: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * multiplier


def trim_labels_to_budget(labels: torch.Tensor, remaining: int) -> Tuple[torch.Tensor, int]:
    """Keep exactly ``remaining`` causal-LM targets in deterministic row-major order.

    ``labels[:, 1:]`` is generally non-contiguous.  Flattening that slice with
    ``reshape`` can therefore allocate a copy; mutating that flattened tensor
    does not necessarily mutate the original labels tensor.  Work on an
    explicit contiguous clone of the shifted targets, mask the surplus targets
    there, and write the shifted block back into a cloned labels tensor.
    """
    if remaining <= 0:
        raise ValueError("remaining must be positive")

    shifted = labels[:, 1:].clone()
    valid = shifted.ne(-100)
    total = int(valid.sum().item())
    if total <= remaining:
        return labels, total

    # torch.nonzero on the 2-D mask returns coordinates in row-major order,
    # making the exact-budget rule deterministic across conditions and seeds.
    valid_positions = torch.nonzero(valid, as_tuple=False)
    drop_positions = valid_positions[remaining:]
    shifted[drop_positions[:, 0], drop_positions[:, 1]] = -100

    trimmed = labels.clone()
    trimmed[:, 1:] = shifted
    kept = int(trimmed[:, 1:].ne(-100).sum().item())
    if kept != remaining:
        raise RuntimeError(
            f"Budget trim invariant failed: kept {kept} instead of {remaining}"
        )
    return trimmed, kept


def _normalize_accumulated_gradients(parameters: Sequence[torch.nn.Parameter], weight: int) -> None:
    if weight <= 0:
        raise ValueError("Accumulation weight must be positive")
    inverse = 1.0 / float(weight)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(inverse)


def train_preference_alignment(
    model,
    loader: DataLoader,
    protocol: Protocol,
    grad_accumulation: int,
    device: torch.device,
) -> Dict[str, float]:
    """One pass with exact pair-weighted gradient accumulation."""
    parameters = [p for _, p in lora_named_params(model)]
    optimizer = AdamW(parameters, lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    target_tokens = sum(
        sum(item["chosen_mask"]) + sum(item["rejected_mask"])
        for item in loader.dataset.items
    )
    processed = 0
    pairs_seen = 0
    loss_sum = 0.0
    accumulation_steps = 0
    accumulation_pairs = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for batch in tqdm(loader, desc="preference alignment", leave=False):
        loss, info = pairwise_ranking_loss(model, batch, device)
        pair_batch = int(batch["pair_batch_size"].item())
        target = int(info["target_tokens"].detach().item())
        # The batch loss is a mean over pairs. Backpropagating loss*pair_batch
        # and dividing once at the optimizer boundary gives the exact mean over
        # all examples in the accumulation window, including a short last batch.
        (loss * pair_batch).backward()
        accumulation_steps += 1
        accumulation_pairs += pair_batch
        processed += target
        pairs_seen += pair_batch
        loss_sum += loss.detach().float().item() * pair_batch
        if accumulation_steps == grad_accumulation:
            _normalize_accumulated_gradients(parameters, accumulation_pairs)
            torch.nn.utils.clip_grad_norm_(parameters, protocol.grad_clip)
            set_optimizer_lr(
                optimizer,
                protocol.learning_rate,
                token_lr_multiplier(processed, target_tokens, protocol.warmup_ratio),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_steps = 0
            accumulation_pairs = 0
    if accumulation_steps:
        _normalize_accumulated_gradients(parameters, accumulation_pairs)
        torch.nn.utils.clip_grad_norm_(parameters, protocol.grad_clip)
        set_optimizer_lr(
            optimizer,
            protocol.learning_rate,
            token_lr_multiplier(processed, target_tokens, protocol.warmup_ratio),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del optimizer
    return {
        "alignment_train_loss": loss_sum / max(pairs_seen, 1),
        "alignment_train_pairs": pairs_seen,
        "alignment_train_target_tokens": processed,
    }


def infinite_loader(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        yield from loader


def train_shift_phase(
    model,
    loader: DataLoader,
    protocol: Protocol,
    runtime: ModelRuntime,
    device: torch.device,
) -> Dict[str, float]:
    """One phase with an exact loss-bearing target-token budget."""
    parameters = [p for _, p in lora_named_params(model)]
    optimizer = AdamW(parameters, lr=protocol.learning_rate, weight_decay=protocol.weight_decay)
    iterator = infinite_loader(loader)
    processed = 0
    loss_token_sum = 0.0
    accumulation_steps = 0
    accumulation_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    progress = tqdm(total=protocol.tokens_per_phase, unit="tok", desc="shift phase", leave=False)
    while processed < protocol.tokens_per_phase:
        batch = next(iterator)
        ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        remaining = protocol.tokens_per_phase - processed
        labels, target_count = trim_labels_to_budget(labels, remaining)
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(
                input_ids=ids,
                attention_mask=attention,
                labels=labels,
                use_cache=False,
            ).loss
        # Transformers' causal-LM loss is a mean over non-ignored targets.
        # Weighting by target_count and normalizing at the step boundary yields
        # the exact token-weighted mean gradient.
        (loss * target_count).backward()
        accumulation_steps += 1
        accumulation_tokens += target_count
        processed += target_count
        loss_token_sum += loss.detach().float().item() * target_count
        progress.update(target_count)
        if accumulation_steps == runtime.shift_grad_accum:
            _normalize_accumulated_gradients(parameters, accumulation_tokens)
            torch.nn.utils.clip_grad_norm_(parameters, protocol.grad_clip)
            set_optimizer_lr(
                optimizer,
                protocol.learning_rate,
                token_lr_multiplier(processed, protocol.tokens_per_phase, protocol.warmup_ratio),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_steps = 0
            accumulation_tokens = 0
    if accumulation_steps:
        _normalize_accumulated_gradients(parameters, accumulation_tokens)
        torch.nn.utils.clip_grad_norm_(parameters, protocol.grad_clip)
        set_optimizer_lr(
            optimizer,
            protocol.learning_rate,
            token_lr_multiplier(processed, protocol.tokens_per_phase, protocol.warmup_ratio),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    progress.close()
    del optimizer, iterator
    if processed != protocol.tokens_per_phase:
        raise RuntimeError("Phase token budget was not exact")
    return {
        "phase_train_loss": loss_token_sum / processed,
        "phase_target_tokens_actual": processed,
    }


@torch.no_grad()
def apply_update_control(
    model,
    pre_vector: torch.Tensor,
    alignment_gradient: torch.Tensor,
    condition: str,
) -> Dict[str, float]:
    raw_post = lora_vector(model)
    raw_update = raw_post - pre_vector
    gradient_norm_sq = torch.dot(alignment_gradient, alignment_gradient)
    raw_harm = torch.dot(alignment_gradient, raw_update)

    if condition == "unconstrained" or gradient_norm_sq.item() == 0:
        controlled = raw_update
        projected = False
        killed_fraction = 0.0
    elif condition == "global_projection":
        if raw_harm.item() > 0:
            controlled = raw_update - (raw_harm / gradient_norm_sq) * alignment_gradient
            projected = True
        else:
            controlled = raw_update
            projected = False
        killed_fraction = 0.0
    elif condition == "coordinate_mortality":
        coordinate_harm = alignment_gradient * raw_update
        kill = coordinate_harm > 0
        controlled = raw_update.masked_fill(kill, 0.0)
        projected = bool(kill.any().item())
        killed_fraction = kill.float().mean().item()
    else:
        raise ValueError(f"Unknown condition: {condition}")

    construction_harm = torch.dot(alignment_gradient, controlled)
    post_vector = pre_vector + controlled
    write_lora_vector(model, post_vector)
    # Re-read the BF16 parameters. The FP32 construction has g^T v <= 0; the
    # stored state can differ only through BF16 rounding. A deterministic upper
    # bound on the dot-product perturbation is computed from unit roundoff.
    actual_controlled = lora_vector(model) - pre_vector
    post_harm = torch.dot(alignment_gradient, actual_controlled)
    unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
    quantization_bound = unit_roundoff * torch.sum(
        alignment_gradient.abs() * post_vector.abs()
    ).item()
    fp32_bound = 8 * torch.finfo(torch.float32).eps * max(
        (alignment_gradient.norm() * actual_controlled.norm()).item(), 1.0
    )
    tolerance = quantization_bound + fp32_bound
    if condition != "unconstrained" and post_harm.item() > max(construction_harm.item(), 0.0) + tolerance:
        raise RuntimeError(
            f"First-order constraint violated beyond computed rounding bound: "
            f"constructed={construction_harm.item()} stored={post_harm.item()} "
            f"bound={tolerance}"
        )

    intervention = actual_controlled - raw_update
    positive_risk = (alignment_gradient * raw_update).clamp_min(0).sum()
    return {
        "raw_first_order_alignment_change": raw_harm.item(),
        "constructed_first_order_alignment_change": construction_harm.item(),
        "post_first_order_alignment_change": post_harm.item(),
        "raw_update_alignment_descent_cosine": cosine(raw_update, -alignment_gradient),
        "post_update_alignment_descent_cosine": cosine(actual_controlled, -alignment_gradient),
        "raw_update_norm": raw_update.norm().item(),
        "post_update_norm": actual_controlled.norm().item(),
        "retained_update_norm_ratio": (actual_controlled.norm() / raw_update.norm().clamp_min(1e-20)).item(),
        "intervention_norm": intervention.norm().item(),
        "coordinate_killed_fraction": killed_fraction,
        "positive_coordinate_risk_fraction": (
            (alignment_gradient * raw_update > 0).float().mean().item()
        ),
        "positive_coordinate_risk_sum": positive_risk.item(),
        "global_projection_activated": float(projected if condition == "global_projection" else False),
        "bf16_quantization_bound": quantization_bound,
        "constraint_tolerance": tolerance,
    }


@torch.no_grad()
def generate_behavior_sample(
    model,
    tokenizer,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
    generation_seed: int,
    output_path: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> None:
    if output_path.exists():
        LOG.info("SKIP generation %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.jsonl")
    if temporary.exists():
        temporary.unlink()

    was_training = model.training
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    context_limit = int(getattr(model.config, "max_position_embeddings", 32768))

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for row in records:
        chosen_length = len(tokenizer(row["chosen"], add_special_tokens=False)["input_ids"])
        rejected_length = len(tokenizer(row["rejected"], add_special_tokens=False)["input_ids"])
        scored.append((max(chosen_length, rejected_length, 1), row))
    scored.sort(key=lambda item: item[0])

    for start in tqdm(range(0, len(scored), batch_size), desc="behavior generation", leave=False):
        chunk = scored[start : start + batch_size]
        prompts = [qwen_prompt_text(tokenizer, row["messages"]) for _, row in chunk]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        ).to(device)
        prompt_width = encoded["input_ids"].shape[1]
        prompt_max = int(encoded["attention_mask"].sum(dim=1).max().item())
        if prompt_max >= context_limit:
            raise RuntimeError(
                f"Behavior prompt length {prompt_max} reaches/exceeds model context {context_limit}"
            )
        reference_max = max(length for length, _ in chunk)
        max_new_tokens = max(1, min(reference_max, context_limit - prompt_max))
        torch.manual_seed(generation_seed + start)
        torch.cuda.manual_seed_all(generation_seed + start)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        continuation = generated[:, prompt_width:]
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        with open(temporary, "a", encoding="utf-8") as handle:
            for (_, row), text in zip(chunk, texts):
                output_row = {
                    **dict(metadata),
                    "id": row["id"],
                    "messages": row["messages"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "generated": text,
                    "max_new_tokens": max_new_tokens,
                }
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
        del encoded, generated, continuation

    os.replace(temporary, output_path)
    model.config.use_cache = False
    model.train(was_training)
    tokenizer.padding_side = previous_padding
