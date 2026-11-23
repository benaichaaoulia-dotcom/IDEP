#!/usr/bin/env python3
"""B4 — confidence-threshold self-training on MARBERT.

Documented reimplementation of Khalifa, Abdul-Mageed & Shaalan (2021),
"Self-Training Pre-Trained Language Models for Zero- and Few-Shot Multi-Dialectal
Arabic Sequence Labeling" — Algorithm 1, threshold selection (official repository unavailable).

Protocol:
  Phase 1: PlainClassifier(MARBERT) fine-tuned on SEED only —
           1,520 steps MADAR / 260 ElHaj (identical to the plain baseline).
  Phase 2: predict POOL with the phase-1 best checkpoint; keep rows with
           max-softmax >= tau (tau in {0.80, 0.90, 0.95}; val2-selected).
           POOL fine labels are never read; pseudo-labels only.
  Phase 3: continue fine-tuning from the phase-1 best checkpoint on the union
           L = SEED + pseudo-labeled POOL for 760 steps MADAR / 130 ElHaj
           (constant batch 32, uniform draws, same lr/optimizer).
Selection: best monitor (val2 / dev) macro-F1 checkpoint over the whole run.
TEST is never touched here.

Usage:
  python train_selftrain.py --config configs/comp_b4_selftrain_<dataset>_s<seed>.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from baselines.base_data import get_views  # noqa: E402
from baselines.train_baseline import (TextDataset, collate, evaluate, set_seed,  # noqa: E402
                                      main as _unused_main)  # noqa: F401
from common.models import PlainClassifier, param_groups_plain  # noqa: E402


def predict_probs(model, texts, tok, device, batch_size=32, max_length=128):
    """Return (probs [N,C], argmax [N]) for a list of texts (no labels needed)."""
    ds = TextDataset(texts, None, tok, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    all_probs, all_argmax = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            p = torch.softmax(logits, dim=-1)
            all_probs.append(p.cpu())
            all_argmax.append(p.argmax(-1).cpu())
    model.train()
    return torch.cat(all_probs).numpy(), torch.cat(all_argmax).numpy()


def train_phase(model, opt, texts, labels, n_steps, device, log_path, tag,
                batch_size=32, seed=42, tok=None, max_length=128):
    """Constant-batch training on (texts, labels) for n_steps; returns best monitor f1."""
    ds = TextDataset(texts, labels, tok, max_length)
    n = len(ds)
    gen = torch.Generator().manual_seed(seed)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    gpu_s = 0.0
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen).tolist()
        batch = collate([ds[i] for i in idx])
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gold = batch["labels"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            ts = time.time()
        opt.zero_grad()
        logits = model(ids, mask)
        loss = ce(logits, gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_s += time.time() - ts
        if (step + 1) % 25 == 0:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"phase": tag, "step": step + 1, "loss": round(loss.item(), 5)}) + "\n")
            print(f"[{tag} step {step+1}/{n_steps}] loss={loss.item():.4f}", flush=True)
    return gpu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps-cap", type=int, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    dataset = cfg.get("dataset", "madar")
    tau = float(cfg["tau"])
    REPO = BASE.parent
    out_dir = REPO / "runs" / "checkpoints" / run_name
    log_dir = REPO / "runs" / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    if not torch.cuda.is_available():
        print("GPU REQUIRED BUT UNAVAILABLE: torch.cuda.is_available() is False", flush=True)
        raise SystemExit(2)
    device = torch.device("cuda:0")
    print(f"[device] resolved={device} gpu={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    views = get_views(dataset, cfg["seed_fraction"])
    classes = views["classes"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    seed_texts = [t for t, _ in views["labeled"]]
    seed_labels = [c for _, c in views["labeled"]]
    mon_texts = [t for t, _ in views["monitor"]]
    mon_labels = [c for _, c in views["monitor"]]
    pool_texts = views["pool"]
    mon_loader = DataLoader(TextDataset(mon_texts, mon_labels, tok, cfg["max_length"]),
                            batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    model = PlainClassifier(cfg["backbone"], len(classes), cfg["dropout"]).to(device)
    opt = torch.optim.Adam(param_groups_plain(model, cfg["lr"]))
    p0 = next(model.parameters())
    assert p0.device.type == device.type, f"param on {p0.device}, expected {device}"

    n_p1 = cfg["phase1_steps"]
    n_p3 = cfg["phase3_steps"]
    if args.steps_cap:
        n_p1 = min(n_p1, args.steps_cap)
        n_p3 = min(n_p3, max(0, args.steps_cap - n_p1))
    log_path = log_dir / "metrics.jsonl"
    open(log_path, "w").close()
    print(f"[data] dataset={dataset} frac={cfg['seed_fraction']} seed={len(seed_texts)} "
          f"monitor={len(mon_texts)} pool={len(pool_texts)} tau={tau}", flush=True)
    print(f"[train] phase1_steps={n_p1} phase3_steps={n_p3}", flush=True)

    gpu_total = 0.0
    # ---- Phase 1: SEED only ----
    gpu_total += train_phase(model, opt, seed_texts, seed_labels, n_p1, device,
                             log_path, "p1", cfg["batch_size"], cfg["seed"],
                             tok=tok, max_length=cfg["max_length"])
    ev = evaluate(model, mon_loader, device, "plain", classes)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "phase1_end", "monitor_macro_f1": ev["macro_f1"],
                            "monitor_accuracy": ev["accuracy"]}) + "\n")
    print(f"[phase1] monitor_macro_f1={ev['macro_f1']:.4f}", flush=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "classes": classes,
                "backbone": cfg["backbone"], "step": n_p1, "phase": 1},
               out_dir / "phase1_best.pt")

    # ---- Phase 2: pseudo-label the POOL (model-generated labels only) ----
    probs, argmax = predict_probs(model, pool_texts, tok, device,
                                  cfg["batch_size"], cfg["max_length"])
    conf = probs.max(axis=1)
    sel = np.where(conf >= tau)[0]
    pseudo_texts = [pool_texts[i] for i in sel]
    pseudo_labels = [int(argmax[i]) for i in sel]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "pseudo_label", "tau": tau, "n_pool": len(pool_texts),
                            "n_selected": len(pseudo_texts),
                            "fraction_selected": round(len(pseudo_texts) / max(1, len(pool_texts)), 5)}) + "\n")
    print(f"[phase2] tau={tau} pseudo_labels={len(pseudo_texts)}/{len(pool_texts)}", flush=True)

    # ---- Phase 3: fine-tune on L = SEED + pseudo ----
    union_texts = seed_texts + pseudo_texts
    union_labels = seed_labels + pseudo_labels
    gpu_total += train_phase(model, opt, union_texts, union_labels, n_p3, device,
                             log_path, "p3", cfg["batch_size"], cfg["seed"] + 1000,
                             tok=tok, max_length=cfg["max_length"])
    ev = evaluate(model, mon_loader, device, "plain", classes)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "phase3_end", "monitor_macro_f1": ev["macro_f1"],
                            "monitor_accuracy": ev["accuracy"]}) + "\n")
    print(f"[phase3] monitor_macro_f1={ev['macro_f1']:.4f}", flush=True)

    # best checkpoint = better of phase1 / phase3 by monitor macro-F1
    best_phase, best_f1 = 3, ev["macro_f1"]
    # recover phase-1 f1 from the log
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("event") == "phase1_end":
            best_phase, best_f1 = 1, r["monitor_macro_f1"]
            break
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("event") == "phase3_end" and r["monitor_macro_f1"] > best_f1:
            best_phase, best_f1 = 3, r["monitor_macro_f1"]
    src = out_dir / ("phase1_best.pt" if best_phase == 1 else "phase3_last.pt")
    if best_phase == 3:
        torch.save({"model_state_dict": model.state_dict(), "config": cfg, "classes": classes,
                    "backbone": cfg["backbone"], "step": n_p1 + n_p3, "phase": 3},
                   out_dir / "phase3_last.pt")
    torch.save(torch.load(src, map_location="cpu", weights_only=False), out_dir / "best.pt")

    final = {
        "run_name": run_name, "target": "selftrain", "dataset": dataset,
        "seed_fraction": cfg["seed_fraction"], "tau": tau,
        "phase1_steps": n_p1, "phase3_steps": n_p3,
        "n_pseudo_labels": len(pseudo_texts),
        "gpu_busy_seconds_step_time_basis": round(gpu_total, 1),
        "gpu_hours_step_time_basis": round(gpu_total / 3600.0, 4),
        "best_phase": best_phase, "best_monitor_macro_f1": best_f1,
        "device": str(device),
    }
    json.dump(final, open(out_dir / "train_summary.json", "w"), indent=2)
    print(f"[done] {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
