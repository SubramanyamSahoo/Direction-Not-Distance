#!/usr/bin/env python3
"""Numerically verify both parameter-free update controls on the H100."""
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(42)
    g = torch.randn(100_000, generator=generator, device=device, dtype=torch.float32)
    u = torch.randn(100_000, generator=generator, device=device, dtype=torch.float32)

    harm = torch.dot(g, u)
    global_update = u if harm <= 0 else u - harm / torch.dot(g, g) * g
    coordinate_update = u.masked_fill(g * u > 0, 0.0)

    scale_global = max((g.norm() * global_update.norm()).item(), 1.0)
    scale_coordinate = max((g.norm() * coordinate_update.norm()).item(), 1.0)
    eps = torch.finfo(torch.float32).eps
    if torch.dot(g, global_update).item() > 64 * eps * scale_global:
        raise RuntimeError("Global projection identity failed")
    if torch.dot(g, coordinate_update).item() > 64 * eps * scale_coordinate:
        raise RuntimeError("Coordinate mortality identity failed")
    print("METHOD MATH CHECK PASSED")


if __name__ == "__main__":
    main()
