from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_memory_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def estimate_flops(model: torch.nn.Module, sample: torch.Tensor) -> float | None:
    try:
        from thop import profile

        macs, params = profile(model, inputs=(sample,), verbose=False)
        return float(macs)
    except Exception:
        return None


def benchmark_torch_model(model: torch.nn.Module, X: np.ndarray, device: str = "cpu", warmup: int = 20, runs: int = 100) -> Dict[str, float | None]:
    model = model.to(device)
    model.eval()
    sample = torch.tensor(X[:1], dtype=torch.float32).reshape(1, -1, 1).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample)
    end = time.perf_counter()

    latency_s = (end - start) / runs
    throughput = 1.0 / latency_s if latency_s > 0 else None
    params = count_parameters(model)
    memory = estimate_model_memory_mb(model)
    flops = estimate_flops(model, sample)

    return {
        "Params_M": params / 1e6,
        "Memory_MB": memory,
        "FLOPs_M": None if flops is None else flops / 1e6,
        "Latency_ms": latency_s * 1000.0,
        "Throughput_flows_per_s": throughput,
    }
