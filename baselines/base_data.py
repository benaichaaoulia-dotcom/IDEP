#!/usr/bin/env python3
"""Unified weak-regime data views for baselines (MADAR + ElHaj).

One interface for both datasets so every baseline trains/evaluates under the
identical protocol:

    views = get_views("madar", 10)
    -> {"labeled": [(text, class_idx)], "monitor": [(text, class_idx)],
        "pool": [text], "test": [(text, class_idx)], "classes": [...]}

MADAR:  monitor = val2 (train-carved).  ElHaj: monitor = dev.
Pool fine labels are WITHHELD in both; pool texts only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common.data import CLASSES, CLASS2IDX, build_views, texts_for  # noqa: E402
from common.elhaj_data import ELHAJ_CLASSES, ELHAJ_CLASS2IDX, build_elhaj_views  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

_ELHAJ_TEXTS = None


def _elhaj_texts() -> list:
    global _ELHAJ_TEXTS
    if _ELHAJ_TEXTS is None:
        from common.elhaj_data import elhaj_text_index
        _ELHAJ_TEXTS = list(elhaj_text_index())
    return _ELHAJ_TEXTS


def get_views(dataset: str, seed_fraction: int) -> dict:
    assert dataset in ("madar", "elhaj"), dataset
    assert seed_fraction in (5, 10, 20), seed_fraction
    if dataset == "madar":
        v = build_views(seed_fraction)
        return {
            "labeled": list(zip(texts_for(v["labeled"]),
                                [CLASS2IDX[r["label"]] for r in v["labeled"]])),
            "monitor": list(zip(texts_for(v["val2"]),
                                [CLASS2IDX[r["label"]] for r in v["val2"]])),
            "pool": texts_for(v["pool"]),
            "test": list(zip(texts_for(v["test"]),
                             [CLASS2IDX[r["label"]] for r in v["test"]])),
            "classes": list(CLASSES),
        }
    v = build_elhaj_views(seed_fraction)
    texts = _elhaj_texts()
    return {
        "labeled": [(texts[r], c) for r, c in v["labeled"]],
        "monitor": [(texts[r], c) for r, c in v["dev"]],
        "pool": [texts[r] for r in v["pool"]],
        "test": [(texts[r], c) for r, c in v["test"]],
        "classes": list(ELHAJ_CLASSES),
    }


def load_corpus26_seed(seed_fraction: int = 10) -> list:
    """Corpus-26 SEED@f rows (auxiliary training resource for B3 only).

    Derived by data_prep/build_corpus26_seed.py with the SAME SPLIT_SEED and
    group-level FNV-1a rank split as the frozen MADAR split.
    Returns [(text, city_label)] in file order.
    """
    path = REPO / "splits" / "corpus26_seed.jsonl"
    assert path.exists(), f"missing {path} — run data_prep/build_corpus26_seed.py first"
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["seed"][str(seed_fraction)]:
                recs.append((r["text"], r["label"]))
    return recs
