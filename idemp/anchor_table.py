#!/usr/bin/env python3
"""Structural anchor table for the pair-row KL term.

The confidence-gated prior-anchored KL term anchors the intra-fiber
responsibility $q$ to an external structural prior $\\tilde\\pi(x)$:

$$\\pi_c(x) = \\frac{\\exp(\\cos(h_0(x), \\mathbf{e}_c)/\\tau_\\pi)}
                    {\\sum_{k \\in S} \\exp(\\cos(h_0(x), \\mathbf{e}_k)/\\tau_\\pi)},
   \\qquad \\tilde\\pi(x) = \\mathrm{clip}(\\pi(x),\\, \\epsilon,\\, 1-\\epsilon)$$

where $h_0(x)$ is the [CLS] feature of $x$ under the frozen initial backbone
(the pretrained MARBERT weights at load time, so there is no drift by construction)
and $\\mathbf{e}_c$ is the frozen label embedding (initial-backbone [CLS] of a fixed
label-description string). $\\tau_\\pi = 0.5$, $\\epsilon = 0.25$ (frozen;
$\\epsilon$ bounds the KL resting point at $|\\Delta| \\le \\ln 3 \\approx 1.0986$).

Precomputed once before the run for every POOL pair row (16,167 x 2 floats at the
@10% regime) and stored as a checksummed lookup table (`splits/anchor_table.pt`
and `splits/anchor_table.sha256`). At step $t$ the anchor for an injected pair row
is a table lookup by pool index, with zero runtime cost and no drift.

Data firewall: built from POOL rows only (no TEST rows, no fine labels; the fiber
partition comes from the pinned coarse-fiber companion). A data artifact, not a
parameter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

# Frozen structural constants.
TAU_PI = 0.5        # anchor sharpening temperature
EPSILON = 0.25
# Frozen label-description strings: the [CLS]
# feature of these strings under the initial backbone is the frozen label
# embedding e_c for fiber member c. No fine-label access; no parameters.
LABEL_STRINGS = {"RAB": "الدارجة الجزائرية", "TUN": "الدارجة التونسية"}
FIBER_ORDER = ("RAB", "TUN")   # column order of the table (index 0 = RAB, 1 = TUN)

SEED_FRACTION = 10  # frozen @10% regime (the run's binding split)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cls_features(model, tokenizer, texts, max_length: int, batch: int, device) -> torch.Tensor:
    """Frozen [CLS] features (n, 768) — eval mode, no grad, max_length padding."""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = model(ids, mask)
            feats.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(feats, dim=0)


def build_anchor_table(model, tokenizer, pool_texts: list[str], pair_positions: list[int],
                       max_length: int = 128, batch: int = 64, device=None) -> dict:
    """Compute the (n_pair, 2) anchor table from the frozen initial backbone.

    `model`: the backbone module at its INITIAL (official) weights, eval mode.
    `pool_texts`: texts of the POOL view (positions aligned to `pair_positions`).
    `pair_positions`: positions within the pool view of the Maghreb pair rows
        (from build_pool_tables — the SAME index space as training draws).
    Returns payload dict (see save_anchor_table). Deterministic: no RNG, pure
    forward pass (bit-reproducible on a fixed device with the pinned cuBLAS
    workspace pin; the shipped table is checksummed and verified at load).
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # frozen label embeddings e_c (initial-backbone [CLS] of the label strings)
    e = _cls_features(model, tokenizer, [LABEL_STRINGS[c] for c in FIBER_ORDER],
                      max_length, 2, device)
    e = torch.nn.functional.normalize(e, dim=-1)
    # frozen initial-backbone [CLS] features h_0(x) of the pair rows
    texts = [pool_texts[p] for p in pair_positions]
    h0 = _cls_features(model, tokenizer, texts, max_length, batch, device)
    h0 = torch.nn.functional.normalize(h0, dim=-1)
    # pi_c(x) = softmax_S(cos(h_0(x), e_c) / tau_pi); clip to [eps, 1-eps]
    cos = h0 @ e.T                       # (n_pair, 2)
    logits = cos / TAU_PI
    m = logits.max(dim=-1, keepdim=True).values
    pi = torch.exp(logits - m) / torch.exp(logits - m).sum(dim=-1, keepdim=True)
    table = pi.clamp(EPSILON, 1.0 - EPSILON)   # for |S|=2 the clip preserves sum=1
    return {
        "table": table.contiguous(),            # (n_pair, 2) float32
        "pool_positions": [int(p) for p in pair_positions],
        "label_strings": dict(LABEL_STRINGS),
        "fiber_order": list(FIBER_ORDER),
        "tau_pi": TAU_PI,
        "epsilon": EPSILON,
        "seed_fraction": SEED_FRACTION,
        "n_rows": int(len(pair_positions)),
    }


def save_anchor_table(payload: dict, out_path: str | Path) -> str:
    """Atomically write the .pt payload + its .sha256 sidecar; return the hash."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    h = sha256_file(out_path)
    out_path.with_name(out_path.name + ".sha256").write_text(h + "\n", encoding="utf-8")
    return h


def load_anchor_table(path: str | Path, expected_seed_fraction: int = SEED_FRACTION) -> dict:
    """Load the frozen anchor table and verify its checksum + invariants.

    Raises on checksum mismatch, shape/range violation, row-sum drift, or a
    seed-fraction mismatch (the table is regime-bound: @10% pool = 16,167 rows).
    Returns the payload with an added `pos2row` index tensor (pool position ->
    compact-table row; -1 for non-pair positions) for O(1) per-step lookup.
    """
    path = Path(path)
    recorded = path.with_name(path.name + ".sha256").read_text().strip()
    actual = sha256_file(path)
    if actual != recorded:
        raise ValueError(f"anchor table checksum mismatch: recorded={recorded} actual={actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    table = payload["table"]
    n_pair = payload["n_rows"]
    if tuple(table.shape) != (n_pair, 2):
        raise ValueError(f"anchor table shape {tuple(table.shape)} != ({n_pair}, 2)")
    if payload["seed_fraction"] != expected_seed_fraction:
        raise ValueError(f"anchor table regime {payload['seed_fraction']}% != "
                         f"expected {expected_seed_fraction}%")
    if not (table.min().item() >= EPSILON - 1e-6 and table.max().item() <= 1.0 - EPSILON + 1e-6):
        raise ValueError(f"anchor table out of [eps, 1-eps]: "
                         f"[{table.min().item():.4f}, {table.max().item():.4f}]")
    row_sum = table.sum(dim=-1)
    if (row_sum - 1.0).abs().max().item() > 1e-5:
        raise ValueError(f"anchor table row sums deviate from 1: max |dev| "
                         f"{(row_sum - 1.0).abs().max().item():.2e}")
    pos2row = torch.full((payload.get("n_pool", -1),), -1, dtype=torch.long) \
        if payload.get("n_pool", -1) > 0 else None
    if pos2row is not None:
        pos2row[torch.tensor(payload["pool_positions"], dtype=torch.long)] = \
            torch.arange(n_pair, dtype=torch.long)
    payload["pos2row"] = pos2row
    payload["sha256"] = actual
    return payload


def anchor_summary(payload: dict) -> str:
    t = payload["table"]
    return json.dumps({
        "n_rows": int(t.shape[0]),
        "pi_rab_mean": round(float(t[:, 0].mean()), 6),
        "pi_rab_min": round(float(t[:, 0].min()), 6),
        "pi_rab_max": round(float(t[:, 0].max()), 6),
        "frac_at_floor": round(float((t <= EPSILON + 1e-6).any(dim=-1).float().mean()), 6),
        "sha256": payload.get("sha256", ""),
    })
