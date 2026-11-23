#!/usr/bin/env python3
"""Unified locked-TEST (or monitor) evaluation for the baselines.

Every TEST read goes through this script: one recorded
final_eval read per frozen config, with a unique eval id.
The report saves macro-F1, accuracy, per-class F1, confusion matrix, RAB<->TUN
pair mass (MADAR), per-sentence max-softmax + ECE (15 bins), and batch-32
inference throughput / p50 latency — all in that single read.

Supports:
  --kind transformer  (grl | plain | selftrain checkpoints from train_baseline.py /
                       train_selftrain.py)
  --kind tfidf        (sklearn joblib from train_tfidf.py)

Usage:
  python eval_final.py --config <cfg.json> --split test --eval-id COMP-c16-... [--ckpt best]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
import sys

sys.path.insert(0, str(REPO))
from baselines.base_data import get_views  # noqa: E402


def ece15(probs: np.ndarray, gold: np.ndarray) -> float:
    """Expected calibration error, 15 equal-width bins, confidence = max softmax."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == gold).astype(float)
    edges = np.linspace(0.0, 1.0, 16)
    e = 0.0
    n = len(conf)
    for i in range(15):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else (conf >= edges[0]) & (conf <= edges[1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def evaluate_transformer(cfg, split, device, ckpt_name):
    import torch
    from torch.utils.data import DataLoader
    from baselines.train_baseline import TextDataset, collate, set_seed
    from common.models import GRLModel, PlainClassifier

    set_seed(cfg["seed"])
    out_dir = REPO / "runs" / "checkpoints" / cfg["run_name"]
    ckpt = torch.load(out_dir / f"{ckpt_name}.pt", map_location=device, weights_only=False)
    classes = ckpt["classes"]
    if cfg.get("target", "plain") == "grl":
        model = GRLModel(cfg["backbone"], len(classes), cfg["dropout"]).to(device)
    else:
        model = PlainClassifier(cfg["backbone"], len(classes), cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    views = get_views(cfg.get("dataset", "madar"), cfg["seed_fraction"])
    key = "test" if split == "test" else "monitor"
    recs = views[key]
    texts = [t for t, _ in recs]
    gold = np.array([c for _, c in recs])

    ds = TextDataset(texts, list(gold), tok, cfg["max_length"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    all_probs, all_gold = [], []
    warmup, latencies = 10, []
    t_start = time.time()
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if n_batches >= warmup:
                torch.cuda.synchronize() if device.type == "cuda" else None
                t0 = time.time()
            if cfg.get("target", "plain") == "grl":
                logits, _ = model(ids, mask, grl_lambda=0.0)
            else:
                logits = model(ids, mask)
            if n_batches >= warmup:
                torch.cuda.synchronize() if device.type == "cuda" else None
                latencies.append(time.time() - t0)
            p = torch.softmax(logits, dim=-1)
            all_probs.append(p.cpu().numpy())
            all_gold.append(batch["labels"].cpu().numpy())
            n_batches += 1
    wall = time.time() - t_start
    probs = np.vstack(all_probs)
    gold = np.concatenate(all_gold)
    pred = probs.argmax(axis=1)
    n = len(gold)
    # throughput: measured-region sentences / measured-region time (post-warmup)
    measured_sents = max(0, n - 32 * warmup)
    measured_time = sum(latencies) if latencies else wall
    throughput = measured_sents / max(1e-9, measured_time)
    return {
        "probs": probs, "gold": gold, "pred": pred, "classes": classes,
        "n_sentences": int(n),
        "throughput_sent_per_s": float(throughput),
        "p50_batch32_latency_s": float(np.percentile(latencies, 50)) if latencies else None,
        "n_batches": n_batches,
    }


def evaluate_tfidf(cfg, split, ckpt_name):
    import joblib
    out_dir = REPO / "runs" / "checkpoints" / cfg["run_name"]
    pipe = joblib.load(out_dir / f"{ckpt_name}.joblib")
    views = get_views(cfg.get("dataset", "madar"), cfg["seed_fraction"])
    key = "test" if split == "test" else "monitor"
    recs = views[key]
    texts = [t for t, _ in recs]
    gold = np.array([c for _, c in recs])
    classes = views["classes"]
    t0 = time.time()
    proba = pipe.predict_proba(texts)
    wall = time.time() - t0
    pred = proba.argmax(axis=1)
    return {
        "probs": proba, "gold": gold, "pred": pred, "classes": classes,
        "n_sentences": int(len(gold)),
        "throughput_sent_per_s": float(len(gold) / max(1e-9, wall)),
        "p50_batch32_latency_s": None,  # CPU vectorized; batch latency not meaningful
        "n_batches": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["test", "monitor"], required=True)
    ap.add_argument("--eval-id", default=None, help="required for split=test")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--kind", choices=["transformer", "tfidf"], default="transformer")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    dataset = cfg.get("dataset", "madar")
    metrics_dir = REPO / "runs" / "metrics" / run_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "transformer":
        import torch
        if not torch.cuda.is_available():
            print("GPU REQUIRED BUT UNAVAILABLE for transformer eval", flush=True)
            raise SystemExit(2)
        device = torch.device("cuda:0")
        r = evaluate_transformer(cfg, args.split, device, args.ckpt)
    else:
        r = evaluate_tfidf(cfg, args.split, args.ckpt)

    classes = r["classes"]
    gold, pred, probs = r["gold"], r["pred"], r["probs"]
    macro_f1 = f1_score(gold, pred, average="macro", zero_division=0)
    acc = accuracy_score(gold, pred)
    p, rc, f1, sup = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)
    per_class = {classes[i]: {"support": int(sup[i]), "precision": round(float(p[i]), 4),
                              "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4)}
                 for i in range(len(classes))}
    cm = confusion_matrix(gold, pred, labels=list(range(len(classes))))
    report = {
        "run_name": run_name, "dataset": dataset, "split": args.split,
        "eval_id": args.eval_id, "kind": args.kind,
        "n_sentences": r["n_sentences"],
        "macro_f1": round(float(macro_f1), 6), "accuracy": round(float(acc), 6),
        "per_class": per_class,
        "confusion_matrix_rows_gold_cols_pred": {classes[i]: {classes[j]: int(cm[i][j]) for j in range(len(classes))} for i in range(len(classes))},
        "ece15": round(ece15(probs, gold), 6),
        "throughput_sent_per_s": round(r["throughput_sent_per_s"], 3),
        "p50_batch32_latency_s": (round(r["p50_batch32_latency_s"], 6)
                                  if r["p50_batch32_latency_s"] is not None else None),
        "ckpt": args.ckpt,
    }
    if dataset == "madar" and "RAB" in classes and "TUN" in classes:
        i_rab, i_tun = classes.index("RAB"), classes.index("TUN")
        r2t, t2r = int(cm[i_rab][i_tun]), int(cm[i_tun][i_rab])
        report["rab_to_tun"] = r2t
        report["tun_to_rab"] = t2r
        report["pair_mass_pct"] = round(100.0 * (r2t + t2r) / max(1, r["n_sentences"]), 4)
    out_path = metrics_dir / f"{args.split}_report.json"
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("run_name", "dataset", "split", "eval_id",
                                             "n_sentences", "macro_f1", "accuracy", "ece15")}, indent=2))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
