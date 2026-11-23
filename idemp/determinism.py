"""Deterministic setup (seed and RNG control) for the IDEP runner.

Sets the Python, NumPy, Torch, and CUDA seeds at startup, plus the deterministic
cuDNN flags. Full RNG-state save and restore lets an explicit resume continue the
exact random stream (epoch shuffles and replacement draws).

SPLIT_SEED is the frozen split identity recorded in every run record. The per-run
`seed` config field drives the in-run RNGs (gen_stream, gen_pool, gen_rem).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

try:
    from loguru import logger as _loguru
except ImportError:  # pragma: no cover - loguru is in the run environment
    import logging
    _loguru = logging.getLogger("idemp.determinism")

SPLIT_SEED = "2026-11-20-RUN-2026-11-20-02"   # frozen split identity

CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_determinism(seed: int) -> dict:
    """Seed every RNG at startup; return the logged settings record.

    Also pins deterministic cuBLAS (``CUBLAS_WORKSPACE_CONFIG``) and enables
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` — the ``warn_only`` flag is
    the documented fallback for ops that lack a deterministic CUDA implementation (e.g. some
    scatter/gather reductions); those emit a warning, not an error, so the run proceeds on
    the deterministic path everywhere a deterministic kernel exists.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    settings = {"seed": seed, "split_seed": SPLIT_SEED,
                "cudnn_deterministic": True, "cudnn_benchmark": False,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "use_deterministic_algorithms": True,
                "cuda_available": torch.cuda.is_available()}
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:  # logged, then re-raised (fail-fast)
        _loguru.opt(exception=True).error("determinism setup failed: {}", e)
        raise
    _loguru.info("determinism: {}", settings)
    return settings


def save_rng_state() -> dict:
    """Full RNG state (torch CPU + all CUDA devices + python + numpy)."""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": ([torch.cuda.get_rng_state(i)
                        for i in range(torch.cuda.device_count())]
                       if torch.cuda.is_available() else []),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng_state(st: dict) -> None:
    """Restore a save_rng_state() payload (bit-exact)."""
    torch.set_rng_state(st["torch_cpu"])
    if torch.cuda.is_available():
        for i, s in enumerate(st["torch_cuda"]):
            torch.cuda.set_rng_state(s, i)
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
