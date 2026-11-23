#!/usr/bin/env python3
"""Evaluate a trained checkpoint on a split (val2 or test) using the
MADAR-DID-Scorer convention (macro-F1 + accuracy via sklearn).

Usage:
  python evaluate.py --config configs/<name>.json --split test --eval-id my-eval-id

TEST reads are recorded with the given eval_id. val2 is train-carved
and not recorded. Per-class breakdown + confusion matrix are written to
metrics/<run_name>/<split>_report.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
from common.data import CLASSES, CLASS2IDX, build_views, texts_for  # noqa: E402
from common.models import GRLModel, PlainClassifier  # noqa: E402
from baselines.train_baseline import TextDataset, collate, set_seed  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["val2", "test"], required=True)
    ap.add_argument("--eval-id", default=None, help="required for split=test")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    out_dir = REPO / "runs" / "checkpoints" / run_name
    metrics_dir = REPO / "runs" / "metrics" / run_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(out_dir / f"{args.ckpt}.pt", map_location=device, weights_only=False)
    if cfg["target"] == "grl":
        model = GRLModel(cfg["backbone"], len(CLASSES), cfg["dropout"]).to(device)
    else:
        model = PlainClassifier(cfg["backbone"], len(CLASSES), cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    views = build_views(cfg["seed_fraction"])
    recs = views[args.split]
    texts = texts_for(recs)
    labels = [CLASS2IDX[r["label"]] for r in recs]
    tok = __import__("transformers").AutoTokenizer.from_pretrained(cfg["backbone"])
    ds = TextDataset(texts, labels, tok, cfg["max_length"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    all_pred, all_gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if cfg["target"] == "grl":
                logits_label, _ = model(ids, mask, grl_lambda=0.0)
            else:
                logits_label = model(ids, mask)
            all_pred.append(logits_label.argmax(-1).cpu())
            all_gold.append(batch["labels"])
    all_pred = torch.cat(all_pred).numpy()
    all_gold = torch.cat(all_gold).numpy()

    macro_f1 = f1_score(all_gold, all_pred, average="macro", zero_division=0)
    acc = accuracy_score(all_gold, all_pred)
    p, r, f1, sup = precision_recall_fscore_support(all_gold, all_pred, average=None, zero_division=0)
    per_class = {CLASSES[i]: {"support": int(sup[i]), "precision": round(float(p[i]), 4),
                              "recall": round(float(r[i]), 4), "f1": round(float(f1[i]), 4)}
                 for i in range(len(CLASSES))}
    cm = confusion_matrix(all_gold, all_pred, labels=list(range(len(CLASSES))))

    report = {
        "run_name": run_name, "split": args.split, "eval_id": args.eval_id,
        "n_sentences": int(len(all_gold)), "n_groups": len({r["group_id"] for r in recs}),
        "macro_f1": round(float(macro_f1), 4), "accuracy": round(float(acc), 4),
        "per_class": per_class,
        "confusion_matrix_rows_gold_cols_pred": {CLASSES[i]: {CLASSES[j]: int(cm[i][j]) for j in range(len(CLASSES))} for i in range(len(CLASSES))},
        "ckpt": str(out_dir / f"{args.ckpt}.pt"),
        "ckpt_step": ckpt.get("step"),
    }
    out_path = metrics_dir / f"{args.split}_report.json"
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("run_name", "split", "eval_id", "n_sentences",
                                             "macro_f1", "accuracy")}, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
