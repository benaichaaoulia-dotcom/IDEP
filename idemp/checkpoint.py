"""IDEP checkpointing.

Checkpoint payload (`last.pt` = latest; `best.pt` = single-best retention on
the trailing-3-mean selection metric): model state + optimizer state + step +
epoch + latest objective (val2/dev macro-F1 + trailing-3 score) + full RNG
state + config sha256. Format:

    <name>.pt             torch.save payload {version, step, epoch,
                                 model_state_dict, optimizer_state_dict,
                                 objective, rng, config_sha256}
    <name>.pt.sha256      hex sha256 of the .pt file (integrity)

Writes are atomic (tmp + os.replace); loads verify the sha256 sidecar and the
payload version/keys BEFORE returning (a corrupt or foreign checkpoint fails
fast, never silently). Resume is EXPLICIT (the runner requires
`--resume <path>`; there is no auto-detection/overwrite of last.pt/best.pt by
a stray launch).
"""
from __future__ import annotations

import hashlib
import os

import torch

CKPT_VERSION = 1
REQUIRED_KEYS = ("version", "step", "epoch", "model_state_dict",
                 "optimizer_state_dict", "objective", "rng", "config_sha256")


class CheckpointError(RuntimeError):
    pass


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: str, model, optimizer, step: int, epoch: int,
                    objective: dict, rng_state: dict, config_sha256: str,
                    extra: dict | None = None) -> str:
    """Atomically write the checkpoint + sha256 sidecar.

    `extra` carries resume-only fields (e.g. the in-progress epoch perm).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"version": CKPT_VERSION, "step": int(step), "epoch": int(epoch),
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "objective": objective, "rng": rng_state,
               "config_sha256": config_sha256}
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    with open(path + ".sha256", "w") as f:
        f.write(_sha256_of(path) + "\n")
    return path


def load_checkpoint(path: str, model, optimizer=None, device=None) -> dict:
    """Load + integrity-verify a checkpoint (fail-fast), then restore.

    Restores model state (and optimizer state when `optimizer` is given) to
    `device` (default: the model params' device). Returns the payload.
    """
    if not os.path.exists(path):
        raise CheckpointError(f"checkpoint not found: {path}")
    sidecar = path + ".sha256"
    if not os.path.exists(sidecar):
        raise CheckpointError(f"sha256 sidecar missing: {sidecar}")
    expect = open(sidecar).read().strip()
    actual = _sha256_of(path)
    if actual != expect:
        raise CheckpointError(
            f"checkpoint integrity FAILED: sha256 {actual} != {expect} ({path})")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in REQUIRED_KEYS:
        if k not in ck:
            raise CheckpointError(f"checkpoint missing key {k!r} ({path})")
    if ck["version"] != CKPT_VERSION:
        raise CheckpointError(
            f"checkpoint version {ck['version']} != supported {CKPT_VERSION} ({path})")
    dev = device if device is not None else next(model.parameters()).device
    model.load_state_dict({k: v.to(dev) for k, v in ck["model_state_dict"].items()})
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    return ck
