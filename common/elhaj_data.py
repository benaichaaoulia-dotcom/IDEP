"""ElHaj 2018 weak-regime data views.

Dataset: M. ElHaj (2018) Arabic dialects classification, CSV release — 5 classes
EGY/NOR/GLF/LAV/MSA, 8,079 train / 1,010 dev / 1,013 test rows (md5-pinned,
`data_prep/build_elhaj_splits.py`).

Regime (frozen): official train/dev/test kept as split skeleton; weak regime
applied to TRAIN only — seed = 5/10/20% of train source groups (exact-normalized
+ Jaccard >= 0.6 dedup), pool = remaining train sentences with fine labels
WITHHELD; dev = validation (monitoring only); test locked (one final_eval read
per arm). Pinned split: `splits/elhaj/sentences.jsonl` + `summary.json`.

All ElHaj fibers are SINGLETONS (no non-injective fiber) => the IDEP regime runs
with K_p = 0: every replaced row is exact-lifted; the partial-label entropy term
and Maghreb pair machinery are inactive (beta/gamma plumbing stays, pair quota 0).

pi/rho: the rule (rho ∝ F1_seed, pi ∝ 1/F1_seed, normalized over the
5 classes) applied to the ElHaj ANCHOR per-class F1 — data-derived values
computed via `elhaj_weights_from_anchor_f1`, not new constants.
Until then the arm configs carry the uniform 1/5 (SPRM-form) default with a
provenance note.
"""
from __future__ import annotations

import csv
import json
import os
from functools import lru_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.environ.get("IDEP_DATA_ROOT", "data"), "elhaj_arabic_dialects_2018")
SPLIT_DIR = os.path.join(REPO, "splits", "elhaj")

ELHAJ_CLASSES = ["EGY", "NOR", "GLF", "LAV", "MSA"]
ELHAJ_CLASS2IDX = {c: i for i, c in enumerate(ELHAJ_CLASSES)}
ELHAJ_NUM_CLASSES = len(ELHAJ_CLASSES)

_CSV_FILES = (
    ("train", os.path.join(DATA_DIR, "arabic_dialects_train.csv")),
    ("dev", os.path.join(DATA_DIR, "arabic_dialects_dev.csv")),
    ("test", os.path.join(DATA_DIR, "arabic_dialects_test.csv")),
)


@lru_cache(maxsize=1)
def load_elhaj_manifest():
    """Pinned split manifest: list of {row, source, group_id, label, seed{fr:bool}}."""
    path = os.path.join(SPLIT_DIR, "sentences.jsonl")
    assert os.path.exists(path), (
        f"missing pinned ElHaj split {path} — run data_prep/build_elhaj_splits.py first")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return tuple(recs)


@lru_cache(maxsize=1)
def elhaj_text_index():
    """row -> raw text (file order train, dev, test; matches manifest row IDs)."""
    texts = []
    for _source, path in _CSV_FILES:
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sentence", "dialect"], f"unexpected header in {path}"
        texts.extend(text for text, _label in rows[1:])
    return tuple(texts)


def build_elhaj_views(seed_fraction: int) -> dict:
    """Weak-regime views at a seed fraction (5/10/20).

    Returns:
      labeled: list[(row_idx, class_idx)]   — SEED train rows (fine labels visible)
      pool:    list[row_idx]                — POOL train rows (fine labels WITHHELD;
                                              all fibers singleton => exact lift = own label,
                                              resolved at draw time from the manifest,
                                              never via a fine-label read of the view)
      dev:     list[(row_idx, class_idx)]   — validation (monitoring only)
      test:    list[(row_idx, class_idx)]   — locked (one final_eval read per arm)
    """
    assert seed_fraction in (5, 10, 20), seed_fraction
    fr = str(seed_fraction)
    recs = load_elhaj_manifest()
    labeled, pool, dev, test = [], [], [], []
    for r in recs:
        if r["source"] == "train":
            if r["seed"][fr]:
                labeled.append((r["row"], ELHAJ_CLASS2IDX[r["label"]]))
            else:
                pool.append(r["row"])
        elif r["source"] == "dev":
            dev.append((r["row"], ELHAJ_CLASS2IDX[r["label"]]))
        else:
            test.append((r["row"], ELHAJ_CLASS2IDX[r["label"]]))
    return {"labeled": labeled, "pool": pool, "dev": dev, "test": test}


def elhaj_pool_labels(pool_rows) -> list:
    """Exact-lifted labels for pool rows (singleton fibers: lift = own fine label).

    Read from the pinned manifest at draw time — the training VIEW never carries
    pool fine labels this is the deterministic lift operator.
    """
    recs = load_elhaj_manifest()
    by_row = {r["row"]: r for r in recs}
    return [ELHAJ_CLASS2IDX[by_row[row]["label"]] for row in pool_rows]


def elhaj_weights_from_anchor_f1(f1: dict) -> tuple:
    """Rule on the ElHaj anchor per-class F1 (data-derived, not new constants).

    rho_c = F1(c) / sum F1 ;  pi_c = (1/F1(c)) / sum (1/F1)   over the 5 classes.
    `f1` maps each of ELHAJ_CLASSES to its anchor per-class F1 (measured at
    the ElHaj run). Returns (pi, rho) as name-keyed dicts (config format).
    """
    missing = [c for c in ELHAJ_CLASSES if c not in f1]
    assert not missing, f"anchor F1 missing classes: {missing}"
    vals = {c: float(f1[c]) for c in ELHAJ_CLASSES}
    assert all(v > 0 for v in vals.values()), f"non-positive anchor F1: {vals}"
    f1sum = sum(vals.values())
    inv = {c: 1.0 / vals[c] for c in ELHAJ_CLASSES}
    invsum = sum(inv.values())
    rho = {c: vals[c] / f1sum for c in ELHAJ_CLASSES}
    pi = {c: inv[c] / invsum for c in ELHAJ_CLASSES}
    return pi, rho
