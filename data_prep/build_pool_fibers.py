#!/usr/bin/env python3
"""Pinned coarse-fiber companion table.

Builds ``splits/pool_fibers.json``: for every universe row, the (fiber, lifted
fine class) pair computed at DATA-PREP time from the deterministic map M
:

    MSA -> MSA, BEI -> Levant, CAI -> Egypt, DOH -> Gulf, RAB -> Maghreb, TUN -> Maghreb

The TRAINING view reads ONLY this table for pool rows and never
reads the fine label of a pool row. The table is
information-identical to a fine-label derivation because M is a bijection
fine->fiber and the singleton partition == the coarse partition (Maghreb rows
are pooled undistinguished). The fine label is read HERE (data prep), where the
full manifest is in scope — never in the training view.

Deterministic: output is a pure function of the frozen splits/sentences.jsonl
manifest + M (no timestamps, no RNG) => byte-identical regeneration. The frozen
manifests (groups.json / sentences.jsonl / summary.json) are NOT modified.
"""
import hashlib
import json
import sys
from pathlib import Path

SPLITS = Path(__file__).resolve().parent.parent / "splits"

# Deterministic map M — coarse region per fine class.
MAP_M = {"MSA": "MSA", "BEI": "Levant", "CAI": "Egypt", "DOH": "Gulf",
         "RAB": "Maghreb", "TUN": "Maghreb"}
FIBER_ORDER = ["MSA", "Levant", "Egypt", "Gulf", "Maghreb"]
FINE_ORDER = ["MSA", "BEI", "CAI", "DOH", "RAB", "TUN"]  # Corpus-6 head order
FIBER2IDX = {f: i for i, f in enumerate(FIBER_ORDER)}
FINE2IDX = {c: i for i, c in enumerate(FINE_ORDER)}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    summary = json.load(open(SPLITS / "summary.json"))
    split_seed = summary["split_seed"]

    # ---- read the frozen manifest (fine labels in scope at data prep only) ----
    recs = []
    with open(SPLITS / "sentences.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    rows = [r["row"] for r in recs]
    assert rows == list(range(len(rows))), "sentences.jsonl must be in row order"

    table = []
    per_fiber = {f: 0 for f in FIBER_ORDER}
    per_fine_in_fiber = {f: {} for f in FIBER_ORDER}
    for r in recs:
        fine = r["label"]
        fiber = MAP_M[fine]
        lifted = fine if fiber != "Maghreb" else None   # exact lift on singleton fibers only
        table.append([FIBER2IDX[fiber], FINE2IDX.get(lifted, -1)])
        per_fiber[fiber] += 1
        per_fine_in_fiber[fiber][fine] = per_fine_in_fiber[fiber].get(fine, 0) + 1

    out = {
        "split_seed": split_seed,
        "generator": "data_prep/build_pool_fibers.py",
        "map_M": MAP_M,
        "fiber_order": FIBER_ORDER,
        "fine_order": FINE_ORDER,
        "n_rows": len(recs),
        "manifest_md5": {
            "sentences.jsonl": _md5(SPLITS / "sentences.jsonl"),
            "groups.json": _md5(SPLITS / "groups.json"),
            "summary.json": _md5(SPLITS / "summary.json"),
        },
        # data-prep diagnostics (fine-label statistics recorded HERE, never read
        # by the training view): per-fiber totals + fine breakdown.
        "per_fiber_total": per_fiber,
        "per_fiber_fine_counts": per_fine_in_fiber,
        # row i (universe row index) -> [fiber_idx, lifted_fine_idx_or_-1]
        "rows": table,
    }
    out_path = SPLITS / "pool_fibers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"[pool-fibers] wrote {out_path} ({len(table)} rows)")
    print(f"[pool-fibers] per_fiber_total={per_fiber}")
    print(f"[pool-fibers] per_fiber_fine_counts={per_fine_in_fiber}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
