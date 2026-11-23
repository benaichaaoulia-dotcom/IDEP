#!/usr/bin/env python3
"""Training for the two SOTA targets under the finalized WSL-DID weak regime.

Usage:
  python train.py --config configs/<name>.json [--steps-cap N]

Config fields (all frozen per run; see configs/):
  target: "grl" | "plain"
  backbone: HF repo id
  dataset: "madar" | "elhaj"   (default "madar")
  seed_fraction: 5 | 10 | 20
  epochs, batch_size, labeled_per_batch (grl only), max_length, dropout
  lr_backbone, lr_label_head, lr_domain_head (grl) / lr (plain)
  adversarial_weight (grl), grl_gamma (grl)
  seed: int
  run_name: str (output dir name)

Determinism: torch/cuda seeds from `seed`, cudnn.deterministic=True.
Device: explicit CUDA device from CUDA_VISIBLE_DEVICES; resolved device + GPU
name printed at start; runtime assertion that params live on the device.
Monitoring: val2 (train-carved) macro-F1/accuracy per epoch; best checkpoint
by val2 macro-F1. TEST is NEVER touched here.
GPU-h basis: step-time (cuda.synchronize around each optimizer step) summed.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(BASE.parent))
from baselines.base_data import get_views  # noqa: E402
from common.models import GRLModel, PlainClassifier, param_groups_grl, param_groups_plain, progressive_lambda  # noqa: E402


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        **({"labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)} if "label" in batch[0] else {}),
    }


@torch.no_grad()
def evaluate(model, loader, device, target, classes):
    model.eval()
    all_pred, all_gold, total_loss, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if target == "grl":
            logits_label, _ = model(ids, mask, grl_lambda=0.0)
        else:
            logits_label = model(ids, mask)
        pred = logits_label.argmax(dim=-1)
        all_pred.append(pred.cpu())
        all_gold.append(batch["labels"].to(device))
        total_loss += ce(logits_label, batch["labels"].to(device)).item()
        n += len(pred)
    all_pred = torch.cat(all_pred).cpu().numpy()
    all_gold = torch.cat(all_gold).cpu().numpy()
    macro_f1 = f1_score(all_gold, all_pred, average="macro", zero_division=0)
    acc = accuracy_score(all_gold, all_pred)
    per_class = {}
    for i, c in enumerate(classes):
        mask_c = all_gold == i
        if mask_c.sum() == 0:
            per_class[c] = {"support": 0, "f1": 0.0}
            continue
        prec = (all_pred[mask_c] == i).sum() / max(1, (all_pred == i).sum())
        rec = (all_pred[mask_c] == i).sum() / mask_c.sum()
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"support": int(mask_c.sum()), "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4), "f1": round(float(f1), 4)}
    model.train()
    return {"macro_f1": float(macro_f1), "accuracy": float(acc),
            "loss": total_loss / max(1, n), "per_class": per_class,
            "preds": all_pred.tolist(), "golds": all_gold.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps-cap", type=int, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    run_name = cfg["run_name"]
    REPO = BASE.parent
    out_dir = REPO / "runs" / "checkpoints" / run_name
    log_dir = REPO / "runs" / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    if not torch.cuda.is_available():
        print("GPU REQUIRED BUT UNAVAILABLE: torch.cuda.is_available() is False", flush=True)
        raise SystemExit(2)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the physical GPU
    print(f"[device] resolved={device} gpu={torch.cuda.get_device_name(0)} "
          f"vram_total={torch.cuda.get_device_properties(0).total_memory // (1024**2)} MiB "
          f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)

    # ---- data views (weak regime; madar or elhaj) ----
    dataset = cfg.get("dataset", "madar")
    views = get_views(dataset, cfg["seed_fraction"])
    classes = views["classes"]
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    labeled_texts = [t for t, _ in views["labeled"]]
    labeled_labels = [c for _, c in views["labeled"]]
    val2_texts = [t for t, _ in views["monitor"]]
    val2_labels = [c for _, c in views["monitor"]]
    pool_texts = views["pool"]

    # Smoke/overfit support: cap the views (same code path as full runs).
    if cfg.get("max_labeled"):
        labeled_texts = labeled_texts[:cfg["max_labeled"]]
        labeled_labels = labeled_labels[:cfg["max_labeled"]]
        pool_texts = pool_texts[: max(cfg["max_labeled"], 64)]
    labeled_ds = TextDataset(labeled_texts, labeled_labels, tok, cfg["max_length"])
    val2_ds = TextDataset(val2_texts, val2_labels, tok, cfg["max_length"])
    val2_loader = DataLoader(val2_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    # ---- model ----
    if cfg["target"] == "grl":
        model = GRLModel(cfg["backbone"], len(classes), cfg["dropout"]).to(device)
        opt = torch.optim.Adam(
            param_groups_grl(model, cfg["lr_backbone"], cfg["lr_label_head"], cfg["lr_domain_head"]),
        )
    else:
        model = PlainClassifier(cfg["backbone"], len(classes), cfg["dropout"]).to(device)
        opt = torch.optim.Adam(param_groups_plain(model, cfg["lr"]))

    # runtime device assertion (GPU script contract)
    p0 = next(model.parameters())
    assert p0.device.type == device.type, f"param on {p0.device}, expected {device}"

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    n_labeled = len(labeled_ds)
    if cfg["target"] == "grl":
        lpb = cfg["labeled_per_batch"]
        steps_per_epoch = math.ceil(n_labeled / lpb)
    else:
        steps_per_epoch = math.ceil(n_labeled / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    if args.steps_cap:
        total_steps = min(total_steps, args.steps_cap)
    print(f"[data] seed_fraction={cfg['seed_fraction']} labeled={n_labeled} "
          f"val2={len(val2_ds)} pool={len(pool_texts)}", flush=True)
    print(f"[train] target={cfg['target']} steps_per_epoch={steps_per_epoch} "
          f"epochs={cfg['epochs']} total_steps={total_steps}", flush=True)

    pool_ds = TextDataset(pool_texts, None, tok, cfg["max_length"]) if cfg["target"] == "grl" else None
    gen_labeled = torch.Generator().manual_seed(cfg["seed"])
    gen_pool = torch.Generator().manual_seed(cfg["seed"] + 1)

    metrics_log = open(log_dir / "metrics.jsonl", "w", encoding="utf-8")
    best_f1, best_step = -1.0, -1
    gpu_busy_s = 0.0
    t_start = time.time()
    global_step = 0
    stop = False

    epoch = 0
    while epoch < cfg["epochs"] and not stop:
        perm = torch.randperm(n_labeled, generator=gen_labeled).tolist()
        for i in range(0, n_labeled, (lpb if cfg["target"] == "grl" else cfg["batch_size"])):
            if global_step >= total_steps:
                stop = True
                break
            idx = perm[i:i + (lpb if cfg["target"] == "grl" else cfg["batch_size"])]
            lab_ids = [labeled_ds[j] for j in idx]
            batch = collate(lab_ids)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            gold = batch["labels"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.time()
            else:
                t0 = None

            opt.zero_grad()
            if cfg["target"] == "grl":
                lam_p = progressive_lambda(global_step, total_steps, cfg["grl_gamma"])
                # unlabeled portion of the batch (pool, fine labels withheld)
                uidx = torch.randint(0, len(pool_texts), (len(idx),), generator=gen_pool).tolist()
                ubatch = collate([pool_ds[j] for j in uidx])
                uids = ubatch["input_ids"].to(device)
                umask = ubatch["attention_mask"].to(device)
                all_ids = torch.cat([ids, uids], dim=0)
                all_mask = torch.cat([mask, umask], dim=0)
                logits_label, logits_domain = model(all_ids, all_mask, lam_p)
                n_lab = len(idx)
                loss_label = ce(logits_label[:n_lab], gold)
                dom_gold = torch.cat([torch.ones(n_lab, device=device),
                                      torch.zeros(len(uidx), device=device)])
                loss_domain = bce(logits_domain, dom_gold)
                loss = loss_label + cfg["adversarial_weight"] * loss_domain
            else:
                logits_label = model(ids, mask)
                loss = ce(logits_label, gold)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if t0 is not None:
                torch.cuda.synchronize()
                gpu_busy_s += time.time() - t0

            global_step += 1
            if global_step % 25 == 0 or global_step == total_steps:
                rec = {"step": global_step, "epoch": epoch + 1, "loss": loss.item()}
                if cfg["target"] == "grl":
                    rec.update({"loss_label": loss_label.item(), "loss_domain": loss_domain.item(),
                                "grl_lambda": round(lam_p, 5)})
                metrics_log.write(json.dumps(rec) + "\n")
                metrics_log.flush()
                print(f"[step {global_step}/{total_steps}] " +
                      " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()), flush=True)
        epoch += 1
        # ---- val2 monitoring (train-carved; NEVER test) ----
        ev = evaluate(model, val2_loader, device, cfg["target"], classes)
        rec = {"step": global_step, "epoch": epoch, "val2_macro_f1": ev["macro_f1"],
               "val2_accuracy": ev["accuracy"], "val2_loss": ev["loss"]}
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()), flush=True)
        if ev["macro_f1"] > best_f1:
            best_f1, best_step = ev["macro_f1"], global_step
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": cfg, "classes": classes, "backbone": cfg["backbone"],
                "step": global_step, "epoch": epoch,
                "val2_macro_f1": ev["macro_f1"], "val2_accuracy": ev["accuracy"],
            }
            torch.save(ckpt, out_dir / "best.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "classes": classes,
                "backbone": cfg["backbone"], "step": global_step, "epoch": epoch},
               out_dir / "last.pt")

    # Overfit check: for capped (smoke/overfit) runs, evaluate on the training
    # subset itself to confirm model capacity (loss -> ~0, acc -> ~1).
    if cfg.get("max_labeled"):
        tr_ds = TextDataset(labeled_texts, labeled_labels, tok, cfg["max_length"])
        tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)
        ev_tr = evaluate(model, tr_loader, device, cfg["target"], classes)
        rec = {"event": "train_subset_overfit", "n": len(labeled_texts),
               "train_macro_f1": ev_tr["macro_f1"], "train_accuracy": ev_tr["accuracy"],
               "train_loss": ev_tr["loss"]}
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()
        print(f"[overfit] {json.dumps(rec)}", flush=True)

    metrics_log.close()

    wall_s = time.time() - t_start
    final = {
        "run_name": run_name, "target": cfg["target"], "dataset": dataset,
        "seed_fraction": cfg["seed_fraction"],
        "total_steps": total_steps, "wall_seconds": round(wall_s, 1),
        "gpu_busy_seconds_step_time_basis": round(gpu_busy_s, 1),
        "gpu_hours_step_time_basis": round(gpu_busy_s / 3600.0, 4),
        "best_val2_macro_f1": best_f1, "best_step": best_step,
        "device": str(device),
    }
    json.dump(final, open(out_dir / "train_summary.json", "w"), indent=2)
    print(f"[done] {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
