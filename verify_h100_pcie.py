#!/usr/bin/env python3
"""Fast preflight for a single 80 GB H100 PCIe instance.

This is not a training smoke test. It checks the low-level CUDA driver, the
PyTorch CUDA binding, BF16 execution, and sufficient VRAM before any model is
downloaded or trained.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys


def main() -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    code = int(cuda.cuInit(0))
    if code != 0:
        raise RuntimeError(
            f"CUDA driver initialization failed: cuInit(0)={code}. "
            "Do not install packages or start training on this instance."
        )

    import numpy as np
    import torch

    if np.__version__ != "1.26.4":
        raise RuntimeError(f"Expected NumPy 1.26.4; found {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA even though cuInit succeeded")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "H100" not in name.upper():
        raise RuntimeError(f"Expected an NVIDIA H100; found {name}")
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / (1024 ** 3)
    if gib < 75:
        raise RuntimeError(f"Expected an 80 GB H100 class device; detected {gib:.1f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")

    x = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    del x, y

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,pci.bus_id", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception:
        smi = "unavailable"

    result = {
        "cuInit": code,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": name,
        "vram_gib": round(gib, 2),
        "nvidia_smi": smi,
        "bf16_matmul": "passed",
    }
    print(json.dumps(result, indent=2))
    print("H100 PCIE PREFLIGHT PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise
