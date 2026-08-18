#!/usr/bin/env python3
"""Verify only the packages used by this project, ignoring unrelated system extras."""
from importlib.metadata import version

EXPECTED = {
    "numpy": "1.26.4",
    "transformers": "4.56.2",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
    "datasets": "4.4.0",
    "huggingface_hub": "0.35.3",
    "safetensors": "0.6.2",
    "sentencepiece": "0.2.1",
    "protobuf": "6.32.1",
    "pandas": "2.3.3",
    "matplotlib": "3.10.6",
    "tqdm": "4.67.1",
}


def main() -> None:
    mismatches = []
    for package, expected in EXPECTED.items():
        found = version(package)
        print(f"{package:20s} {found}")
        if found != expected:
            mismatches.append((package, expected, found))
    if mismatches:
        raise RuntimeError(f"Package version mismatch: {mismatches}")
    import torch
    import transformers
    import peft
    import datasets
    print("torch                ", torch.__version__)
    print("torch CUDA runtime   ", torch.version.cuda)
    print("PROJECT PYTHON STACK PASSED")


if __name__ == "__main__":
    main()
