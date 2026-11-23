#!/usr/bin/env python3
"""One-time pre-run build of the structural anchor table.

Loads the OFFICIAL initial MARBERT backbone (pinned local HF cache), computes
the frozen label embeddings e_c and the per-pool-row anchors pi_tilde(x) for
all 16,167 Maghreb pair rows of the frozen @10% POOL split, and stores the
checksummed table at splits/anchor_table.pt (+ .sha256 sidecar).

Data firewall: POOL rows only (no TEST rows, no fine labels; the fiber partition
is the pinned coarse-fiber companion). No RNG is consumed (pure forward pass),
so the build is deterministic on a fixed device; the shipped table is verified
by checksum at every training start (load_anchor_table).

Usage:
  uv run ... python idemp/build_anchor_table.py [--out splits/anchor_table.pt]
          [--batch 64] [--max-length 128] [--gpu -1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


import torch  # noqa: E402

from common.data import build_pool_tables, build_views, texts_for  # noqa: E402
from common.models import PlainClassifier  # noqa: E402
from idemp.anchor_table import (SEED_FRACTION, anchor_summary, build_anchor_table,  # noqa: E402
                                save_anchor_table)
from idemp.determinism import set_determinism  # noqa: E402
from idemp.resources import resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "splits" / "anchor_table.pt"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=-1, help="-1 = auto-select (least-loaded)")
    args = ap.parse_args()

    set_determinism(20261120)   # fixed seed for the build (no RNG is consumed; parity pin)
    device = resolve_device(args.gpu, allow_cpu=False,
                            record_path=str(Path(args.out).parent / "anchor_build_gpu.json"))
    print(f"[build] device={device} gpu={torch.cuda.get_device_name(device)}", flush=True)

    views = build_views(SEED_FRACTION)
    pool = views["pool"]
    _by_class, pair_pos = build_pool_tables(pool)
    pool_texts = texts_for(pool)
    print(f"[build] pool={len(pool)} pair_rows={len(pair_pos)} "
          f"(positions within the pool view; same index space as training draws)", flush=True)

    # Official initial backbone (frozen state = weights at load time).
    model = PlainClassifier("UBC-NLP/MARBERT", 6, 0.5).bert
    model = model.to(device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")

    t0 = time.time()
    payload = build_anchor_table(model, tok, pool_texts, pair_pos,
                                 max_length=args.max_length, batch=args.batch, device=device)
    payload["n_pool"] = len(pool)
    payload["backbone"] = "UBC-NLP/MARBERT"
    h = save_anchor_table(payload, args.out)
    print(f"[build] done in {time.time() - t0:.1f}s -> {args.out}", flush=True)
    print(f"[build] sha256={h}", flush=True)
    print(f"[build] summary={anchor_summary(payload)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
