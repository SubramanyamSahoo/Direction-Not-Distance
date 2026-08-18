#!/usr/bin/env python3
"""Cache exact Hub revisions and write an immutable asset manifest.

No training is performed. The manifest records commit SHAs so every subsequent
run loads the same model and dataset revisions, including in offline mode.
"""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

CONFIG = Path("config.json")
MANIFEST = Path("asset_manifest.json")


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    p = cfg["protocol"]
    model_ids = [v["model_id"] for v in cfg["models"].values()] + [p["judge_model"]]
    dataset_ids = [p["hh_dataset"], p["owt_dataset"]]

    api = HfApi()
    manifest = {"models": {}, "datasets": {}}

    for repo_id in model_ids:
        info = api.model_info(repo_id)
        if not info.sha:
            raise RuntimeError(f"No commit SHA returned for model {repo_id}")
        manifest["models"][repo_id] = info.sha
        print(f"Caching model {repo_id}@{info.sha}")
        snapshot_download(repo_id=repo_id, repo_type="model", revision=info.sha)

    for repo_id in dataset_ids:
        info = api.dataset_info(repo_id)
        if not info.sha:
            raise RuntimeError(f"No commit SHA returned for dataset {repo_id}")
        manifest["datasets"][repo_id] = info.sha
        print(f"Caching dataset repository {repo_id}@{info.sha}")
        snapshot_path = Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", revision=info.sha)
        )
        if repo_id == p["hh_dataset"]:
            for split_name in ("train", "test"):
                required = snapshot_path / p["hh_config"] / f"{split_name}.jsonl.gz"
                if not required.is_file():
                    raise FileNotFoundError(f"Missing required HH-RLHF file: {required}")
        elif repo_id == p["owt_dataset"]:
            parquet_files = list((snapshot_path / "data").glob("*.parquet"))
            if not parquet_files:
                parquet_files = list(snapshot_path.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No Parquet files found in cached dataset snapshot: {snapshot_path}")

    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(MANIFEST)
    print(json.dumps(manifest, indent=2))
    print("ASSET PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
