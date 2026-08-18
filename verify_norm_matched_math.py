#!/usr/bin/env python3
import torch

torch.manual_seed(42)
for n in (1, 17, 4096):
    g = torch.randn(n, dtype=torch.float32)
    u = torch.randn(n, dtype=torch.float32)
    cm = u.masked_fill(g * u > 0, 0.0)
    raw_norm = u.norm()
    r = torch.tensor(1.0) if raw_norm.item() == 0 else cm.norm() / raw_norm
    nm = r * u
    err = abs(nm.norm().item() - cm.norm().item())
    bound = 16 * torch.finfo(torch.float32).eps * max(raw_norm.item(), cm.norm().item(), 1.0)
    assert err <= bound, (n, err, bound)
    if u.norm().item() > 0 and nm.norm().item() > 0:
        cosine = torch.dot(u, nm) / (u.norm() * nm.norm())
        assert cosine.item() > 1 - 32 * torch.finfo(torch.float32).eps
print("NORM-MATCHED MATH CHECK PASSED")
