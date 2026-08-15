"""Positive control for compute-sanitizer: does memcheck actually see this process?

The grouped run finished in the same 10 s under every tool as it does bare, which is not
what an instrumented kernel usually costs. Either the GPU work is a small share of a
mostly-import-bound test session, or the sanitizer attached to a process that launched no
kernels. A clean report is worth nothing until the difference is settled, so: launch a
kernel that is definitely wrong, under the same command line, and require the tool to say
so. If this prints 0 errors the earlier 0 errors mean nothing.
"""

import torch

x = torch.arange(64, device="cuda", dtype=torch.float32)
idx = torch.tensor([1 << 20], device="cuda", dtype=torch.long)
try:
    y = torch.index_select(x, 0, idx)
    torch.cuda.synchronize()
    print("OOB gather returned", y.item())
except Exception as exc:  # noqa: BLE001
    print("OOB gather raised:", type(exc).__name__)
