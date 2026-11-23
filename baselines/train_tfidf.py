#!/usr/bin/env python3
"""B5 — TF-IDF + L2 logistic regression.

N-gram/linear family for Arabic dialect identification (Eltanbouly et al., MADAR
shared task 2019 — word + char n-grams; Liakata et al. 2017 — SVM char n-grams,
foundational). Protocol:
  features: word unigrams + char n-grams 2-5, sublinear_tf=True, lowercase=False,
            min_df=2, max_features=200_000 (FeatureUnion)
  model:    LogisticRegression(C=1.0, solver='liblinear', random_state=seed)
  train:    SEED@f rows only (weak regime). Deterministic => seed-invariant
            (3 seeds produce identical models).

Usage:
  python train_tfidf.py --config configs/comp_b5_tfidf_<dataset>_s<seed>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

BASE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(BASE.parent))
from baselines.base_data import get_views  # noqa: E402


def build_pipeline(random_state: int) -> FeatureUnion:
    word = TfidfVectorizer(lowercase=False, ngram_range=(1, 1), analyzer="word",
                           sublinear_tf=True, min_df=2, max_features=200_000)
    char = TfidfVectorizer(lowercase=False, ngram_range=(2, 5), analyzer="char",
                           sublinear_tf=True, min_df=2, max_features=200_000)
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("feat", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(C=1.0, solver="liblinear", random_state=random_state)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    dataset = cfg.get("dataset", "madar")
    REPO = BASE.parent
    out_dir = REPO / "runs" / "checkpoints" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    views = get_views(dataset, cfg["seed_fraction"])
    texts = [t for t, _ in views["labeled"]]
    labels = np.array([c for _, c in views["labeled"]])
    print(f"[data] dataset={dataset} frac={cfg['seed_fraction']} n_seed={len(texts)} "
          f"n_classes={len(views['classes'])}", flush=True)

    t0 = time.time()
    pipe = build_pipeline(cfg["seed"])
    pipe.fit(texts, labels)
    wall = time.time() - t0

    # train-set sanity (not a selection signal; monitor split used for reporting only)
    mon_texts = [t for t, _ in views["monitor"]]
    mon_labels = np.array([c for _, c in views["monitor"]])
    from sklearn.metrics import f1_score, accuracy_score
    p = pipe.predict(mon_texts)
    rec = {
        "run_name": run_name, "target": "tfidf_logreg", "dataset": dataset,
        "seed_fraction": cfg["seed_fraction"], "seed": cfg["seed"],
        "n_seed_rows": len(texts), "n_classes": len(views["classes"]),
        "classes": views["classes"],
        "monitor_macro_f1": float(f1_score(mon_labels, p, average="macro", zero_division=0)),
        "monitor_accuracy": float(accuracy_score(mon_labels, p)),
        "fit_seconds": round(wall, 2), "device": "cpu",
    }
    json.dump(rec, open(out_dir / "train_summary.json", "w"), indent=2)
    joblib_path = out_dir / "best.joblib"
    import joblib
    joblib.dump(pipe, joblib_path)
    print(f"[done] {json.dumps(rec)}", flush=True)
    print(f"[saved] {joblib_path}", flush=True)


if __name__ == "__main__":
    main()
