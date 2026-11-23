#!/usr/bin/env python3
"""B3 — HierarchicalArabicDialectID (Baimukan, Bouamor & Habash, LREC 2022).

Official repository: CAMeL-Lab/HierarchicalArabicDialectID, used read-only via
HIER_ARABIC_DIR. Protocol:
  main model:  26-class city TF-IDF (word 1-gram + char 1-3-gram) with OVR-NB,
               trained on Corpus-26 SEED@10% (same split seed and group split).
  extra model: 6-class TF-IDF with OVR-NB on Corpus-6 SEED@f; its predict_proba is
               appended to the main feature vector.
  LM features: kenlm word and char LMs (.arpa.binary), loaded from B3_LM_ROOT.
  aggregated layers: country and region LayerObjects when the LMs are present;
               otherwise a recorded reduction.
  ElHaj: extra-model-only variant (no Corpus-26 auxiliary; recorded reduction).
  Evaluation: one TEST read, macro-F1 (sklearn convention), per-class and confusion.
Deterministic (TF-IDF + NB): seed-invariant.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, os.path.join(os.environ.get("HIER_ARABIC_DIR", "hierarchical_arabic_dialect_id"), "classification"))
sys.path.insert(0, os.path.join(os.environ.get("HIER_ARABIC_DIR", "hierarchical_arabic_dialect_id"), "data_preprocess"))

from baselines.base_data import get_views, load_corpus26_seed  # noqa: E402
from standardize_label import city_map  # noqa: E402

LM_ROOT = Path(os.environ.get("B3_LM_ROOT", "runs/b3_lm/binary"))
WORK = REPO / "runs" / "b3_work"

# ElHaj 2018 has 5 dialect classes with no city-level labels; bridge each to
# the nearest MADAR city code inside B3's native label space (documented
# protocol decision, recorded in the test report):
ELHAJ_TO_CITY = {
    "EGY": "CAI",   # Egyptian -> Cairo (nile_basin)
    "NOR": "AMM",   # North Levantine -> Amman (Jordan)
    "GLF": "DOH",   # Gulf -> Doha (Qatar)
    "LAV": "BEI",   # Southern Levantine -> Beirut (Lebanon)
    "MSA": "MSA",
}


def city_to_parts(code: str):
    """3-letter city code -> (city, country, region) via the official map."""
    v = city_map.get(code.lower())
    assert v, f"city_map missing code {code}"
    return v[0], v[2], v[3]


def make_lm_links(label: str, kind: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    city, country, region = city_to_parts(label)
    src = LM_ROOT / "city" / kind / f"{city}-{country}-{region}.arpa.binary"
    assert src.exists(), f"missing LM {src}"
    dst = dest_dir / f"{label}.arpa"
    if not dst.exists():
        dst.symlink_to(src)


def build_tsv(path: Path, rows):
    """rows: [(text, city_code)] -> TSV in the official repo format
    (dialect_city_id = lowercase city name, per madar_shared_task1.ipynb)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("original_sentence\tdialect_city_id\tdialect_country_id\tdialect_region_id\n")
        for text, code in rows:
            city, country, region = city_to_parts(code)
            f.write(f"{text}\t{city}\t{country}\t{region}\n")


def ece15(probs, gold):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == gold).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
    e = 0.0
    n = len(conf)
    for i in range(15):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="train + evaluate on the monitor split only (no TEST read)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    dataset = cfg.get("dataset", "madar")
    assert dataset in ("madar", "elhaj"), f"unknown dataset {dataset}"

    out_dir = REPO / "runs" / "checkpoints" / run_name
    metrics_dir = REPO / "runs" / "metrics" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    views = get_views(dataset, cfg["seed_fraction"])
    classes6 = views["classes"]

    def to_city(cls_name: str) -> str:
        if dataset == "madar":
            return cls_name  # MADAR classes ARE the city codes
        return ELHAJ_TO_CITY[cls_name]

    seed6 = [(t, to_city(classes6[c])) for t, c in views["labeled"]]  # (text, city_code)
    test_texts = [t for t, _ in views["test"]]
    gold = np.array([c for _, c in views["test"]])  # idx into classes6

    aux_rows = None
    if dataset == "madar":
        aux_rows = load_corpus26_seed(cfg["seed_fraction"])  # [(text, city_code)]

    WORK.mkdir(exist_ok=True)
    tsv_main = WORK / f"{run_name}_main.tsv"
    tsv_extra = WORK / f"{run_name}_extra.tsv"
    tsv_test = WORK / f"{run_name}_test.tsv"
    if aux_rows:
        build_tsv(tsv_main, aux_rows)
    else:
        build_tsv(tsv_main, seed6)  # extra-model-only variant
    build_tsv(tsv_extra, seed6)
    build_tsv(tsv_test, [(t, to_city(classes6[c])) for t, c in views["test"]])
    print(f"[data] main rows={len(aux_rows) if aux_rows else len(seed6)} "
          f"(aux={bool(aux_rows)}) extra rows={len(seed6)} test rows={len(test_texts)}", flush=True)

    # ---------- LM mapping (city LMs for all 26 ADIDA labels) ----------
    from classifier import DialectIdentifier, ADIDA_LABELS, ADIDA_LABELS_EXTRA
    char_dir = WORK / "lm_mapped" / "char"
    word_dir = WORK / "lm_mapped" / "word"
    for lab in ADIDA_LABELS:
        make_lm_links(lab, "char", char_dir)
        make_lm_links(lab, "word", word_dir)

    layers = []
    layer_status = {}
    for level in ("country", "region"):
        cdir = LM_ROOT / level / "char"
        wdir = LM_ROOT / level / "word"
        n_c = len(list(cdir.glob("*.arpa.binary"))) if cdir.exists() else 0
        n_w = len(list(wdir.glob("*.arpa.binary"))) if wdir.exists() else 0
        ok = n_c >= 3 and n_w >= 3  # region has only 6 LMs; country has 22
        layer_status[level] = {"char_lms": n_c, "word_lms": n_w, "used": bool(ok)}
        if ok:
            from utils import LayerObject
            ddir = WORK / f"lm_mapped_{level}"
            (ddir / "lm" / "char").mkdir(parents=True, exist_ok=True)
            (ddir / "lm" / "word").mkdir(parents=True, exist_ok=True)
            for p in cdir.glob("*.arpa.binary"):
                l = ddir / "lm" / "char" / (p.stem + ".arpa")
                if not l.exists():
                    l.symlink_to(p)
            for p in wdir.glob("*.arpa.binary"):
                l = ddir / "lm" / "word" / (p.stem + ".arpa")
                if not l.exists():
                    l.symlink_to(p)
            layers.append(LayerObject({
                "level": level, "data_dir": str(ddir), "kenlm_train": False,
                "kenlm_train_files": [], "exclude_list": [],
                "train_path": str(tsv_main),  # LayerObject.train() wraps in a list
                "use_lm": True, "use_distr": True,
                "cols_train": ["dialect_country_id", "dialect_region_id"]
                               if level == "country" else ["dialect_region_id"],
            }))
    print(f"[layers] {json.dumps(layer_status)}", flush=True)

    # ---------- train (official repo code path) ----------
    t0 = time.time()
    d = DialectIdentifier(labels=ADIDA_LABELS, labels_extra=ADIDA_LABELS_EXTRA,
                          char_lm_dir=str(char_dir), word_lm_dir=str(word_dir),
                          aggregated_layers=layers or None,
                          result_file_name=str(WORK / f"{run_name}_results.json"),
                          extra=True)
    d.train(data_path=[str(tsv_main)], data_extra_path=[str(tsv_extra)])
    train_s = time.time() - t0
    print(f"[train] done in {train_s:.1f}s", flush=True)

    # ---------- evaluation read ----------
    if args.smoke:
        split = "monitor"
        mon = views["monitor"]  # (text, class_idx); unlimited reads (train-carved / dev)
        texts = [t for t, _ in mon]
        gold = np.array([c for _, c in mon])
        eval_id = None
    else:
        split = "test"
        eval_id = cfg["eval_id"]
        texts = test_texts

    # combined label space of the main model (sorted, per LabelEncoder)
    main_codes = ADIDA_LABELS if aux_rows else {to_city(c) for c in classes6}
    combined = sorted({f"{city_to_parts(c)[0]}-{city_to_parts(c)[1]}-{city_to_parts(c)[2]}"
                       for c in main_codes})
    idx6 = [combined.index(f"{city_to_parts(to_city(c))[0]}-{city_to_parts(to_city(c))[1]}-{city_to_parts(to_city(c))[2]}")
            for c in classes6]

    # batch-latency probe (32-sentence batches, official predict path)
    lat = []
    for i in range(0, len(texts), 32):
        b = texts[i:i + 32]
        t0 = time.time()
        d._prepare_sentences(b)
        lat.append(time.time() - t0)
    lat = np.array(lat)

    t0 = time.time()
    x_final = d._prepare_sentences(texts)
    proba_full = d._classifier.predict_proba(x_final)
    assert proba_full.shape[1] == len(combined), \
        f"prob shape {proba_full.shape} != label space {len(combined)}"
    proba6 = proba_full[:, idx6]
    pred6 = proba6.argmax(axis=1)
    eval_s = time.time() - t0

    macro_f1 = f1_score(gold, pred6, average="macro", zero_division=0)
    acc = accuracy_score(gold, pred6)
    p, rc, f1, sup = precision_recall_fscore_support(gold, pred6, average=None, zero_division=0)
    per_class = {classes6[i]: {"support": int(sup[i]), "precision": round(float(p[i]), 4),
                               "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4)}
                 for i in range(len(classes6))}
    cm = confusion_matrix(gold, pred6, labels=list(range(len(classes6))))
    report = {
        "run_name": run_name, "dataset": dataset, "split": split,
        "eval_id": eval_id, "kind": "b3_hierarchical",
        "n_sentences": int(len(gold)),
        "macro_f1": round(float(macro_f1), 6), "accuracy": round(float(acc), 6),
        "per_class": per_class,
        "confusion_matrix_rows_gold_cols_pred": {classes6[i]: {classes6[j]: int(cm[i][j]) for j in range(len(classes6))} for i in range(len(classes6))},
        "ece15": round(ece15(proba6, gold), 6),
        "throughput_sent_per_s": round(len(gold) / max(1e-9, eval_s), 3),
        "p50_batch32_latency_s": round(float(np.percentile(lat, 50)), 4),
        "train_seconds": round(train_s, 1), "eval_seconds": round(eval_s, 1),
        "aggregated_layers": layer_status,
        "n_main_rows": len(aux_rows) if aux_rows else len(seed6),
        "n_extra_rows": len(seed6),
    }
    if dataset == "elhaj":
        report["elhaj_bridge_mapping"] = ELHAJ_TO_CITY
    if dataset == "madar" and "RAB" in classes6 and "TUN" in classes6:
        i_rab, i_tun = classes6.index("RAB"), classes6.index("TUN")
        r2t, t2r = int(cm[i_rab][i_tun]), int(cm[i_tun][i_rab])
        report["rab_to_tun"] = r2t
        report["tun_to_rab"] = t2r
        report["pair_mass_pct"] = round(100.0 * (r2t + t2r) / max(1, len(gold)), 4)
    out_path = metrics_dir / f"{split}_report.json"
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("run_name", "dataset", "macro_f1", "accuracy", "ece15")}, indent=2))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
