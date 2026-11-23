#!/usr/bin/env python3
"""Data views for the WSL-DID weak regime (MADAR Corpus-6, group-level splits).

Views for a seed fraction f (5/10/20):
  - labeled:   SEED@f groups minus val2@f  (text + fine label)
  - val2:      val2@f groups               (text + fine label; train-carved monitoring ONLY)
  - pool:      POOL@f groups               (text only; fine labels WITHHELD)
  - test:      TEST groups                 (text + fine label; locked, final eval only)

Class set (Corpus-6): MSA, BEI, CAI, DOH, RAB, TUN — fixed order for all heads.
"""
import os
import json
from pathlib import Path

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"

# Fixed class order (Corpus-6). Deterministic index mapping.
CLASSES = ["MSA", "BEI", "CAI", "DOH", "RAB", "TUN"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_sentences():
    recs = []
    with open(SPLITS_DIR / "sentences.jsonl", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs



FIBER_ORDER = ["MSA", "Levant", "Egypt", "Gulf", "Maghreb"]
# deterministic map M: fine class -> coarse region
MAP_M = {"MSA": "MSA", "BEI": "Levant", "CAI": "Egypt", "DOH": "Gulf",
         "RAB": "Maghreb", "TUN": "Maghreb"}

_FIBERS_CACHE = None


def load_pool_fibers():
    """row -> (fiber_name, lifted_class_name|None) from the pinned companion.

    The table is computed at data prep from the deterministic map M (fine label
    in scope there ONLY). The training view consumes this table — it never
    reads a pool row's fine label.
    """
    global _FIBERS_CACHE
    if _FIBERS_CACHE is None:
        with open(SPLITS_DIR / "pool_fibers.json", encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["n_rows"] == len(load_sentences()), "pool_fibers.json row count mismatch"
        _FIBERS_CACHE = {
            "obj": obj,
            "table": [(FIBER_ORDER[f], CLASSES[c] if c >= 0 else None)
                      for f, c in obj["rows"]],
        }
    return _FIBERS_CACHE["table"]


def build_pool_tables(view_recs):
    """Pool index tables from the pinned coarse-fiber table ONLY.

    view_recs: the POOL view records (any seed fraction). NO ``label`` field of
    any record is read — the partition is a lookup in the pinned fiber table.

    Returns (pool_idx_by_class, pool_pair_idx):
      pool_idx_by_class: dict class-idx -> list[int] of positions within the
                         view (exact-lifted singleton-fiber rows);
      pool_pair_idx: list[int] of positions within the view (Maghreb pair rows,
                     admissible set {RAB,TUN}; undistinguished).
    """
    table = load_pool_fibers()
    by_class = {c: [] for c in range(len(CLASSES))}
    pair = []
    for pos, rec in enumerate(view_recs):
        fiber, lifted = table[rec["row"]]
        if fiber == "Maghreb":
            pair.append(pos)
        else:
            by_class[CLASS2IDX[lifted]].append(pos)
    return by_class, pair


def build_views(seed_fraction):
    """Return dict of views for the given seed fraction (5, 10, or 20)."""
    assert seed_fraction in (5, 10, 20), seed_fraction
    recs = load_sentences()
    labeled, val2, pool, test = [], [], [], []
    for r in recs:
        text = None  # text is not stored in sentences.jsonl; recover below if needed
        if r["is_test"]:
            test.append(r)
        elif r["seed"][str(seed_fraction)]:
            if r.get("val2", {}).get(str(seed_fraction), False):
                val2.append(r)
            else:
                labeled.append(r)
        else:
            pool.append(r)
    return {"labeled": labeled, "val2": val2, "pool": pool, "test": test}


def texts_for(view_recs):
    """Recover raw sentence texts for a list of sentence records (row -> text)."""
    rows = [r["row"] for r in view_recs]
    text_by_row = load_text_index()
    return [text_by_row[i] for i in rows]


_TEXT_CACHE = None


def load_text_index():
    """row index -> raw sentence text (universe order train-then-dev)."""
    global _TEXT_CACHE
    if _TEXT_CACHE is None:
        idx = {}
        base = Path(os.path.join(os.environ.get("IDEP_DATA_ROOT", "data"), "madar_shared_task_2019/MADAR-Shared-Task-Subtask-1"))
        i = 0
        for name in ("MADAR-Corpus-6-train.tsv", "MADAR-Corpus-6-dev.tsv"):
            with open(base / name, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    text, _label = line.split("\t")
                    idx[i] = text
                    i += 1
        _TEXT_CACHE = idx
    return _TEXT_CACHE
