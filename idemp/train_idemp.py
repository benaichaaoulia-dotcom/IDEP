#!/usr/bin/env python3
"""Training harness for IDEP (inverted-decoupling partial-label pool-row mixing).

Each step uses a constant-size batch. Seed rows get exact supervised cross-entropy
with logit adjustment. Pool rows are injected as singleton-fiber rows (class weights
inverse to seed F1) and as Maghreb pair-pool rows (admissible set {RAB, TUN}); pair
rows use a confidence-gated, prior-anchored KL term and a logit-margin partial-label
marginal. The pair-row anchor is read from a precomputed lookup table, so the
per-step cost is dominated by the anchor lookup.

Setting lambda_kl = lambda_m = 0 reduces the pair-row term exactly to the
partial-label marginal baseline. Run settings come from a JSON config (see
idemp/runconfig.py for the fields).

Usage:
  uv run python idemp/train_idemp.py --config idemp/configs/<name>.json
"""
import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

SCREEN = Path(__file__).resolve().parent
REPO = SCREEN.parent
import sys
sys.path.insert(0, str(REPO))

from common.data import CLASSES, CLASS2IDX, build_pool_tables, build_views, texts_for  # noqa: E402
from common.models import PlainClassifier  # noqa: E402
from idemp.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from idemp.determinism import SPLIT_SEED, restore_rng_state, save_rng_state, set_determinism  # noqa: E402
from idemp.resources import (FirstStepWatchdog, GpuMonitor, apply_cpu_cap,  # noqa: E402
                             assert_params_on_device, resolve_device)
from idemp.runlog import init_runlog, write_step_event, write_summary_atomic  # noqa: E402

CYCLE_MARK = os.environ.get("IDEMP_CYCLE_MARK", "IDEP")

KS_DEFAULT = 12                # singleton replacement rows per full batch; config-overridable
KP_DEFAULT = 4                 # Maghreb-pair replacement rows per full batch; config-overridable
EARLY_STOP_STEP = 760
EARLY_STOP_FLOOR = 0.8712
ENGAGE_STEPS = (50, 152)
SEL_TIE_TOL = 1e-4             # trailing-mean selection tie tolerance (latest wins within tol)
SINGLETON_FIBERS = (CLASS2IDX["MSA"], CLASS2IDX["BEI"], CLASS2IDX["CAI"], CLASS2IDX["DOH"])
MAGHREB_FIBERS = (CLASS2IDX["RAB"], CLASS2IDX["TUN"])
SEED_F1 = {"MSA": 93.46, "BEI": 87.67, "CAI": 86.17, "DOH": 82.14}
_inv = {c: 1.0 / v for c, v in SEED_F1.items()}
_norm = sum(_inv.values())
PI_WEIGHTS = {c: _inv[c] / _norm for c in SEED_F1}   # injection weights (DOH-heaviest)
# INVERTED removal weights (NEW — proportional to seed F1, the deferral-inversion):
# MSA 93.46/349.44=.2675, BEI 87.67/349.44=.2509, CAI 86.17/349.44=.2466, DOH 82.14/349.44=.2351
# => strongest class (MSA, saturated) removed MOST, weakest (DOH) removed LEAST.
_f1sum = sum(SEED_F1.values())
RHO_WEIGHTS = {c: SEED_F1[c] / _f1sum for c in SEED_F1}   # removal weights (INVERTED — deferral inversion)
GAMMA = 1.0   # logit-adjustment strength (batch-prior form), on the 28 exactly-supervised rows
LAMBDA_KL = 0.5   # anchored-KL weight, on the 4 Maghreb pair rows (initial)
LAMBDA_M = 0.1    # logit-margin weight, on the 4 Maghreb pair rows (initial)
TAU_PI = 0.5      # anchor sharpening temperature (anchor-table build)
EPS_ANCHOR = 0.25
GAMMA_G = 2.0     # coarse-gate exponent: g_coarse = (p_a + p_b)^gamma_g
TAU_M = 1.0       # margin threshold (decisiveness floor on |z_RAB - z_TUN|)



def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def tokenize_bank(texts, tok, max_length):
    enc = tok(list(texts), truncation=True, max_length=max_length, padding="max_length")
    return torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"])


@torch.no_grad()
def evaluate(model, ids, mask, gold, device, classes=None, class2idx=None):
    """Plain argmax evaluation (no calibration; inference = fine head argmax).

    `classes`/`class2idx`: dataset class list (MADAR-6 by default; ElHaj-5 when
    passed) — the RAB<->TUN confusion block is computed only when both classes
    exist (MADAR; ElHaj has no Maghreb pair fiber).
    """
    if classes is None:
        classes, class2idx = CLASSES, CLASS2IDX
    model.eval()
    all_pred, total_loss, n = [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    bs = 64
    for i in range(0, len(gold), bs):
        z = model(ids[i:i + bs].to(device), mask[i:i + bs].to(device))
        all_pred.append(z.argmax(dim=-1).cpu())
        total_loss += ce(z, gold[i:i + bs].to(device)).item()
        n += len(gold[i:i + bs])
    all_pred = torch.cat(all_pred).cpu().numpy()
    gold_np = gold.numpy()
    macro_f1 = f1_score(gold_np, all_pred, average="macro", zero_division=0)
    per_class = {}
    for c in classes:
        i = class2idx[c]
        mask_c = gold_np == i
        if mask_c.sum() == 0:
            per_class[c] = {"support": 0, "f1": 0.0}
            continue
        prec = (all_pred[mask_c] == i).sum() / max(1, (all_pred == i).sum())
        rec = (all_pred[mask_c] == i).sum() / mask_c.sum()
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"support": int(mask_c.sum()), "precision": round(float(prec), 4),
                        "recall": round(float(rec), 4), "f1": round(float(f1), 4)}
    if "RAB" in class2idx and "TUN" in class2idx:
        i_rab, i_tun = class2idx["RAB"], class2idx["TUN"]
        rab_to_tun = int(((gold_np == i_rab) & (all_pred == i_tun)).sum())
        tun_to_rab = int(((gold_np == i_tun) & (all_pred == i_rab)).sum())
        pair_block = {"rab_to_tun": rab_to_tun, "tun_to_rab": tun_to_rab,
                      "pair_mass_pct": round(100.0 * (rab_to_tun + tun_to_rab) / max(1, n), 4)}
    else:
        pair_block = {"rab_to_tun": None, "tun_to_rab": None, "pair_mass_pct": None}
    model.train()
    return {"macro_f1": float(macro_f1), "accuracy": float(accuracy_score(gold_np, all_pred)),
            "loss": total_loss / max(1, n), "per_class": per_class,
            "n_total": int(n), **pair_block}


def draw_class(weights_by_class, gen):
    """Draw a class index from the weight dict (keys = class idx). Deterministic on gen."""
    keys = sorted(weights_by_class.keys())
    w = torch.tensor([weights_by_class[k] for k in keys], dtype=torch.float32)
    w = w / w.sum()
    j = int(torch.multinomial(w, 1, generator=gen).item())
    return keys[j]


def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max-shifted log-sum-exp (no underflow)."""
    m = x.max(dim=dim, keepdim=True).values
    return (m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))).squeeze(dim)


class _MarginRegFn(torch.autograd.Function):
    """Squared-hinge intra-fiber logit-gap regularizer.

    $$\\mathrm{MarginReg}(z, S) = \\tfrac{1}{2}\\max\\big(0,\\, \\tau_m - |z_a - z_b|\\big)^2$$

    Analytic derivative w.r.t. $\\Delta = z_a - z_b$:
    $-\\max(0, \\tau_m - |\\Delta|) \\cdot \\mathrm{sign}(\\Delta)$, with the
    deterministic tie-breaker $\\mathrm{sign}(0) := +1$ at the measure-zero
    $\\Delta = 0$ saddle (under the autodiff
    $\\mathrm{sign}(0) = 0$ convention the hinge has a zero-gradient saddle;
    in practice any generic gradient from the marginal/CE terms breaks the tie,
    but the implementation must not rely on that). For $|\\Delta| \\ge \\tau_m$
    the term and its gradient are exactly 0.
    """

    @staticmethod
    def forward(ctx, delta: torch.Tensor, tau_m: float) -> torch.Tensor:
        u = (tau_m - delta.abs()).clamp_min(0.0)
        ctx.save_for_backward(delta, u)
        return 0.5 * u * u

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        delta, u = ctx.saved_tensors
        sgn = torch.where(delta < 0, -1.0, 1.0)
        return grad_output * (-u) * sgn, None     # None for the non-tensor tau_m


def partial_label_anch_loss(z_pair: torch.Tensor, pi_tilde: torch.Tensor,
                            i_rab: int, i_tun: int, lambda_kl: float,
                            lambda_m: float) -> torch.Tensor:
    """Confidence-Gated Prior-Anchored KL + Logit-Margin partial-label term
.

    $$L_{\\text{partial}}^{\\text{anch}}(x, S) = -\\log \\sum_{c \\in S} p_c(x)
       + \\lambda_{KL}\\, g(x)\\, D_{KL}(\\tilde\\pi(x) \\| q(\\cdot \\mid x, S))
       + \\lambda_m\\, g_{\\text{coarse}}(x)\\, \\mathrm{MarginReg}(z(x), S),
       \\; S = \\{\\text{RAB}, \\text{TUN}\\}$$

    Components (all on existing forward-pass quantities; zero new parameters):
      * marginal = logsumexp(z) - logsumexp(z_S) with max-shifted reductions
        (exactly -log sum_{c in S} p_c); clamp_min(0.0) is only an O(eps_mach)
        guard at the zero-loss boundary.
      * q in LOGIT SPACE: Delta = z_RAB - z_TUN, log q_a = -softplus(-Delta),
        log q_b = -softplus(Delta) — finite across the full float32 range, no
        0/0 as p_a + p_b -> 0.
      * D_KL(pi_tilde || q) = sum_c pi_tilde_c log(pi_tilde_c / q_c); pi_tilde
        is the frozen precomputed anchor so
        log pi_tilde is finite; gradient w.r.t. Delta is q_a - pi_tilde_a,
        bounded in [-1, 1].
      * gates: g_coarse = sg[(p_a + p_b)^gamma_g],
        g = sg[1 - max_c p_c] * g_coarse — the directional KL is gated by the
        full g (never overrides a confident prediction); the margin is gated by
        g_coarse alone (it sharpens the existing lean, imposes no direction).
      * MarginReg via _MarginRegFn (squared hinge, sign(0):=+1 tie-breaker).
    lambda_kl = lambda_m = 0 reduces EXACTLY to the SPRM marginal
    baseline. z_pair: (N, C) UNADJUSTED logits of the pair rows; pi_tilde:
    (N, 2) frozen anchor rows aligned with z_pair. Returns per-row loss (N,).
    """
    lse_all = logsumexp(z_pair, dim=-1)
    lse_fiber = logsumexp(z_pair[:, [i_rab, i_tun]], dim=-1)
    marginal = (lse_all - lse_fiber).clamp_min(0.0)
    if lambda_kl == 0.0 and lambda_m == 0.0:
        return marginal
    delta = z_pair[:, i_rab] - z_pair[:, i_tun]
    p = torch.softmax(z_pair, dim=-1)
    p_fiber = p[:, [i_rab, i_tun]]
    g_coarse = p_fiber.sum(dim=-1).pow(GAMMA_G).detach()          # sg[(p_a + p_b)^gamma_g]
    g = ((1.0 - p.max(dim=-1).values).clamp_min(0.0) * g_coarse).detach()  # sg[1-max_c p_c]*g_coarse
    loss = marginal
    if lambda_kl > 0.0:
        log_q_a = -F.softplus(-delta)   # = log sigma(delta)
        log_q_b = -F.softplus(delta)    # = log sigma(-delta)
        pi_a, pi_b = pi_tilde[:, 0], pi_tilde[:, 1]
        kl = (pi_a * torch.log(pi_a) + pi_b * torch.log(pi_b)
              - pi_a * log_q_a - pi_b * log_q_b)
        loss = loss + lambda_kl * g * kl
    if lambda_m > 0.0:
        loss = loss + lambda_m * g_coarse * _MarginRegFn.apply(delta, TAU_M)
    return loss


def resolve_weights(cfg_weights, defaults, name, class_names=None):
    """Config-overridable class weights; frozen defaults when absent.

    cfg_weights: None -> frozen defaults (module constants, exact 6-dp values);
    dict {class-name: weight} -> validated (keys exactly the dataset's singleton
    classes — MADAR's four by default, ElHaj's five via `class_names` — weights
    > 0, sum 1 +/- 1e-6). Returns dict class-idx -> float.
    """
    if class_names is None:
        class_names = [CLASSES[c] for c in SINGLETON_FIBERS]
        idx_of = CLASS2IDX
    else:
        idx_of = {c: i for i, c in enumerate(class_names)}
    if cfg_weights is None:
        src = defaults
    else:
        expect_keys = set(class_names)
        if set(cfg_weights.keys()) != expect_keys:
            raise ValueError(f"{name}: keys {sorted(cfg_weights)} != {sorted(expect_keys)}")
        if any(v <= 0 for v in cfg_weights.values()):
            raise ValueError(f"{name}: all weights must be > 0")
        s = sum(cfg_weights.values())
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"{name}: weights sum {s!r} != 1 (+/- 1e-6)")
        src = cfg_weights
    return {idx_of[k]: float(v) for k, v in src.items()}


def draw_replacement_sprpm(batch_labels, pool_idx_by_class, pool_pair_idx,
                           KS, KP, gen_rem, gen_pool, rho_weights, pi_weights,
                           sing_fibers=None, mag_fibers=None):
    """Per-step IDEP replacement (D7.3.3; SPRM K=16 axis).

    batch_labels: tensor of fine labels of the current seed batch (len B).
    pool_idx_by_class: dict class-idx -> tensor of singleton-pool row indices.
    pool_pair_idx: tensor of Maghreb-pair pool row indices.
    rho_weights / pi_weights: dict class-idx -> weight (config-overridable;
    frozen defaults => bit-identical draws to the screen harness).
    sing_fibers / mag_fibers: dataset fiber tuples (frozen MADAR defaults;
    ElHaj passes all-5-singletons + empty pair fiber).
    Returns (removal_positions: list[int], sing_pool_rows: tensor[KS'], sing_labels: tensor[KS'],
             pair_pool_rows: tensor[KP']).
    KS' = min(KS, #singleton-fiber rows in batch); KP' = min(KP, #Maghreb rows in batch).
    Deterministic on (gen_rem, gen_pool). Arm-independent.
    """
    if sing_fibers is None:
        sing_fibers, mag_fibers = SINGLETON_FIBERS, MAGHREB_FIBERS
    sing_pos = [i for i, y in enumerate(batch_labels.tolist()) if y in sing_fibers]
    mag_pos = [i for i, y in enumerate(batch_labels.tolist()) if y in mag_fibers]
    Ksp = min(KS, len(sing_pos))
    Kpp = min(KP, len(mag_pos))

    # ---- removal: Ks' singleton seed rows with INVERTED class weights rho (deferral control) ----
    pos_by_class = {c: [i for i in sing_pos if batch_labels[i].item() == c] for c in sing_fibers}
    removals = []
    used = set()
    for _ in range(Ksp):
        avail = {c: pos for c, pos in pos_by_class.items() if any(p not in used for p in pos)}
        if not avail:
            break
        w = {c: rho_weights[c] for c in avail}
        c = draw_class(w, gen_rem)
        cands = [p for p in pos_by_class[c] if p not in used]
        r = int(torch.randint(0, len(cands), (1,), generator=gen_rem).item())
        removals.append(cands[r])
        used.add(cands[r])

    # ---- removal: Kp' Maghreb seed rows uniform over {RAB,TUN} (ElHaj: none) ----
    mag_by_class = {c: [i for i in mag_pos if batch_labels[i].item() == c] for c in mag_fibers}
    for _ in range(Kpp):
        avail = {c: pos for c, pos in mag_by_class.items() if any(p not in used for p in pos)}
        if not avail:
            break
        w = {c: 0.5 for c in avail}
        c = draw_class(w, gen_rem)
        cands = [p for p in mag_by_class[c] if p not in used]
        r = int(torch.randint(0, len(cands), (1,), generator=gen_rem).item())
        removals.append(cands[r])
        used.add(cands[r])

    # ---- injection: Ks' singleton pool rows with class weights pi (with replacement) ----
    sing_rows, sing_labels = [], []
    for _ in range(len(removals) - Kpp if len(removals) > Kpp else 0):
        pass  # placeholder (counting handled below via Ksp/Kpp)
    # singleton injections = Ksp (matches singleton removals)
    for _ in range(Ksp):
        c = draw_class({c: pi_weights[c] for c in sing_fibers}, gen_pool)
        rows = pool_idx_by_class[c]
        r = int(torch.randint(0, len(rows), (1,), generator=gen_pool).item())
        sing_rows.append(int(rows[r]))
        sing_labels.append(c)

    # ---- injection: Kp' pair pool rows uniform over {RAB,TUN} (with replacement) ----
    pair_rows = []
    for _ in range(Kpp):
        r = int(torch.randint(0, len(pool_pair_idx), (1,), generator=gen_pool).item())
        pair_rows.append(int(pool_pair_idx[r]))

    return (removals, torch.tensor(sing_rows, dtype=torch.long),
            torch.tensor(sing_labels, dtype=torch.long),
            torch.tensor(pair_rows, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps-cap", type=int, default=None)
    ap.add_argument("--no-test", action="store_true", help="skip the final TEST eval (smoke/infra runs)")
    ap.add_argument("--smoke-eval", action="store_true",
                    help="run one val2 (held-out probe) evaluation at the end of a short smoke run")
    ap.add_argument("--resume", type=str, default=None, metavar="PATH",
                    help="EXPLICIT resume from a checkpoint (no auto-detection; "
                         "a new run starts clean)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="explicit CPU path for preflight checks (without it, no CUDA "
                         "fails fast — no silent CPU fallback)")
    args = ap.parse_args()

    global GAMMA, LAMBDA_KL, LAMBDA_M
    cfg = json.load(open(args.config))
    arm = cfg["arm"]
    assert arm in ("a0", "a2", "a3"), arm
    run_name = cfg["run_name"]
    KS = int(cfg.get("k_singleton", KS_DEFAULT))
    KP = int(cfg.get("k_pair", KP_DEFAULT))
    GAMMA = float(cfg.get("gamma", GAMMA))   # config-overridable (diagnostics)
    if "beta" in cfg:
        raise ValueError(f"{run_name}: config field 'beta' is retired — "
                         f"use 'lambda_kl'/'lambda_m'")
    LAMBDA_KL = float(cfg.get("lambda_kl", LAMBDA_KL))
    LAMBDA_M = float(cfg.get("lambda_m", LAMBDA_M))
    LAMBDA_CONTRAST = float(cfg.get("lambda_contrast", 0.0))
    CONTRAST_MARGIN = float(cfg.get("contrast_margin", 1.0))
    # ElHaj 2018 independent second dataset:
    # 5 classes EGY/NOR/GLF/LAV/MSA, ALL singleton fibers => KP must be 0 and
    # the pair-fiber machinery (entropy term, pair draws) is inactive. MADAR-6
    # is the default and its behavior is unchanged (frozen constants intact).
    dataset = cfg.get("dataset", "madar")
    if dataset == "elhaj":
        from common.elhaj_data import (ELHAJ_CLASSES, ELHAJ_CLASS2IDX, build_elhaj_views,
                                       elhaj_pool_labels, elhaj_text_index)
        CLASSES_, CLASS2IDX_ = ELHAJ_CLASSES, ELHAJ_CLASS2IDX
        assert KP == 0, "ElHaj regime: all fibers singleton => k_pair must be 0"
        PATH_NOTE = "ElHaj 2018 train-carved weak regime"
    else:
        assert dataset == "madar", f"unknown dataset {dataset!r}"
        CLASSES_, CLASS2IDX_ = CLASSES, CLASS2IDX
        PATH_NOTE = "MADAR Corpus-6 group-level"

    PI_W = resolve_weights(cfg.get("pi_weights"), PI_WEIGHTS, "pi_weights", class_names=CLASSES_)
    RHO_W = resolve_weights(cfg.get("rho_weights"), RHO_WEIGHTS, "rho_weights", class_names=CLASSES_)
    EARLY_STOP_STEP_ = int(cfg.get("early_stop_step", EARLY_STOP_STEP))
    EARLY_STOP_FLOOR_ = cfg.get("early_stop_floor", EARLY_STOP_FLOOR)   # None disables
    ENGAGE_STEPS_ = tuple(cfg.get("engage_steps", list(ENGAGE_STEPS)))
    K = KS + KP
    out_dir = SCREEN / "checkpoints" / run_name
    log_dir = SCREEN / "logs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(cfg["seed"])
    apply_cpu_cap(float(cfg.get("cpu_cap_fraction", 0.8)))

    allow_cpu = bool(cfg.get("allow_cpu", False)) or args.allow_cpu
    device = resolve_device(int(cfg.get("gpu", 0)), allow_cpu=allow_cpu,
                            record_path=str(log_dir / "gpu_selection.json"))
    print(f"[device] resolved={device} "
          f"gpu={torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'} "
          f"allow_cpu={allow_cpu}", flush=True)

    tb_writer, wb_run = init_runlog(str(log_dir), run_name)
    gpu_mon = GpuMonitor(str(log_dir), run_name, dev=device)
    gpu_mon.start()

    if dataset == "madar":
        views = build_views(cfg["seed_fraction"])
        seed_texts = texts_for(views["labeled"])
        seed_labels = torch.tensor([CLASS2IDX_[r["label"]] for r in views["labeled"]], dtype=torch.long)
        pool_texts = texts_for(views["pool"])
        mon_view, mon_split = views["val2"], "val2"
    else:  # elhaj: pinned weak-regime views (dev = validation; test locked)
        views = build_elhaj_views(cfg["seed_fraction"])
        texts = elhaj_text_index()
        seed_texts = [texts[r] for r, _c in views["labeled"]]
        seed_labels = torch.tensor([_c for _r, _c in views["labeled"]], dtype=torch.long)
        pool_texts = [texts[r] for r in views["pool"]]
        mon_view, mon_split = views["dev"], "dev"

    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    mon_texts = texts_for(mon_view) if dataset == "madar" else [elhaj_text_index()[r] for r, _c in mon_view]
    mon_ids, mon_mask = tokenize_bank(mon_texts, tok, cfg["max_length"])
    mon_labels = torch.tensor([CLASS2IDX_[r["label"]] for r in mon_view]
                              if dataset == "madar" else [_c for _r, _c in mon_view], dtype=torch.long)

    n_seed = len(seed_labels)
    seed_ids, seed_mask = tokenize_bank(seed_texts, tok, cfg["max_length"])
    pool_ids, pool_mask = tokenize_bank(pool_texts, tok, cfg["max_length"])
    n_pool = len(views["pool"])

    if dataset == "madar":
        pool_by_class, pool_pair_pos = build_pool_tables(views["pool"])
        sing_fibers = SINGLETON_FIBERS
    else:
        pool_by_class = {i: [] for i in range(len(CLASSES_))}
        for _pos, (_r, _lab) in enumerate(zip(views["pool"], elhaj_pool_labels(views["pool"]))):
            pool_by_class[_lab].append(_pos)
        pool_pair_pos = []
        sing_fibers = tuple(range(len(CLASSES_)))
    pool_idx_by_class = {c: torch.tensor(pool_by_class[c], dtype=torch.long) for c in sing_fibers}
    n_sing_pool = sum(len(v) for v in pool_idx_by_class.values())
    pool_pair_idx = torch.tensor(pool_pair_pos, dtype=torch.long)
    n_pair_pool = len(pool_pair_idx)
    anchor = None
    if dataset == "madar" and KP > 0:
        from idemp.anchor_table import load_anchor_table
        anchor_path = cfg.get("anchor_table", str(REPO / "splits" / "anchor_table.pt"))
        anchor = load_anchor_table(anchor_path,
                                   expected_seed_fraction=cfg["seed_fraction"])
        tbl_positions = anchor["pool_positions"]
        assert len(tbl_positions) == n_pair_pool, \
            f"anchor table rows {len(tbl_positions)} != pool pair rows {n_pair_pool}"
        assert tbl_positions == list(pool_pair_idx.tolist()), \
            "anchor table pool positions != build_pool_tables pair positions (index-space drift)"
        assert anchor["pos2row"] is not None and int(anchor["pos2row"].max()) == n_pair_pool - 1
        print(f"[anchor] table loaded: rows={n_pair_pool} sha256={anchor['sha256'][:16]}… "
              f"tau_pi={anchor['tau_pi']} eps={anchor['epsilon']} (checksum verified, POOL-only)",
              flush=True)
    print(f"[data] dataset={dataset} seed_fraction={cfg['seed_fraction']} seed={n_seed} pool={n_pool} "
          f"singleton_pool={n_sing_pool} (" + "/".join(CLASSES_) + ")=" +
          json.dumps({CLASSES_[c]: int(len(pool_idx_by_class[c])) for c in sing_fibers}) +
          f" pair_pool={n_pair_pool} (Maghreb, undistinguished)", flush=True)
    print(f"[idemp] arm={arm} KS={KS} KP={KP} K={K} pi(inject)=" +
          json.dumps({CLASSES_[c]: round(PI_W[c], 6) for c in sing_fibers}) +
          " rho(remove)=INVERTED ~seedF1 " +
          json.dumps({CLASSES_[c]: round(RHO_W[c], 6) for c in sing_fibers}) +
          f" gamma(logitadj)={GAMMA} lambda_kl(anchored)={LAMBDA_KL} lambda_m(margin)={LAMBDA_M} "
          f"[tau_pi={TAU_PI} eps={EPS_ANCHOR} gamma_g={GAMMA_G} tau_m={TAU_M}]",
          flush=True)

    model = PlainClassifier(cfg["backbone"], len(CLASSES_), cfg["dropout"]).to(device)
    assert_params_on_device(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    print(f"[opt] single_lr={cfg['lr']:.2e} (backbone+head, one Adam, one LR)", flush=True)
    watchdog = FirstStepWatchdog(float(cfg.get("max_first_step_seconds", 600.0)))

    steps_per_epoch = math.ceil(n_seed / cfg["batch_size"])   # anchor parity (no drop_last)
    total_steps = steps_per_epoch * cfg["epochs"]
    if args.steps_cap:
        total_steps = min(total_steps, args.steps_cap)
    n_repl_total = total_steps * K  # upper bound (last partial batches use K')
    print(f"[train] arm={arm} stream_rows/epoch={n_seed} steps_per_epoch={steps_per_epoch} "
          f"epochs={cfg['epochs']} total_steps={total_steps} KS={KS} KP={KP} "
          f"pool_row_uses_upper_bound={n_repl_total}", flush=True)

    # Arm-independent generators => a2/a3 batch identity (twin parity).
    gen_stream = torch.Generator().manual_seed(cfg["seed"])      # stream shuffle (anchor parity)
    gen_pool = torch.Generator().manual_seed(cfg["seed"] + 1)    # injection draws
    gen_rem = torch.Generator().manual_seed(cfg["seed"] + 2)     # removal draws

    cfg_sha = json.dumps({k: v for k, v in cfg.items()}, sort_keys=True, default=str)
    config_sha256 = hashlib.sha256(cfg_sha.encode()).hexdigest()

    metrics_log = open(log_dir / "metrics.jsonl", "w", encoding="utf-8")
    val2_hist = []
    best_score, best_step, best_f1 = -1.0, -1, -1.0
    gpu_busy_s = 0.0
    t_start = time.time()
    global_step = 0
    start_epoch, start_i = 0, 0
    stop = False
    early_stopped = False

    # EXPLICIT resume: restore model+optimizer+
    # RNGs (global + the three arm generators) and continue from the recorded
    # step/epoch offset. A new run starts clean.
    RESUMED_PERM = None
    if args.resume:
        ck = load_checkpoint(args.resume, model, opt, device=device)
        restore_rng_state(ck["rng"]["global"])
        gen_stream.set_state(ck["rng"]["gen_stream"])
        gen_pool.set_state(ck["rng"]["gen_pool"])
        gen_rem.set_state(ck["rng"]["gen_rem"])
        global_step = ck["step"]
        start_epoch = ck["epoch"]
        start_i = (global_step % steps_per_epoch) * cfg["batch_size"]
        RESUMED_PERM = ck.get("epoch_perm")
        obj = ck["objective"]
        val2_hist = list(obj.get("val2_hist", []))
        best_score, best_step, best_f1 = obj["best_score"], obj["best_step"], obj["best_f1"]
        print(f"[resume] EXPLICIT from {args.resume}: step={global_step} "
              f"epoch={start_epoch} i={start_i} best_score={best_score}", flush=True)
    n_repl_rows_total = 0

    ce_none = nn.CrossEntropyLoss(reduction="none")
    i_rab, i_tun = MAGHREB_FIBERS   # ElHaj: unused (KP=0, no pair rows)

    epoch = start_epoch
    while epoch < cfg["epochs"] and not stop:
        # resume: reuse the checkpointed perm for the in-progress epoch (the
        # generator state was saved AFTER that perm was drawn); at an epoch
        # boundary (start_i == 0) the next perm is drawn fresh from the
        # restored generator (correct stream position)
        if epoch == start_epoch and start_i > 0 and RESUMED_PERM is not None:
            perm = RESUMED_PERM
        else:
            perm = torch.randperm(n_seed, generator=gen_stream).tolist()

        for i in range(start_i if epoch == start_epoch else 0, n_seed, cfg["batch_size"]):
            if global_step >= total_steps:
                stop = True
                break
            idx = perm[i:i + cfg["batch_size"]]
            idx_t = torch.tensor(idx)
            ids = seed_ids[idx_t].to(device)
            mask = seed_mask[idx_t].to(device)
            gold = seed_labels[idx_t].to(device)

            # ---- SPRM replacement (a2/a3 only; a0 = pure anchor stream) ----
            if arm in ("a2", "a3"):
                removals, sing_rows, sing_labels, pair_rows = draw_replacement_sprpm(
                    seed_labels[idx_t], pool_idx_by_class, pool_pair_idx, KS, KP, gen_rem, gen_pool,
                    RHO_W, PI_W, sing_fibers=sing_fibers,
                    mag_fibers=MAGHREB_FIBERS if dataset == "madar" else ())
                keep = [p for p in range(len(idx)) if p not in set(removals)]
                keep_t = torch.tensor(keep, dtype=torch.long)
                if len(removals) > 0:
                    ids = torch.cat([ids[keep_t], pool_ids[sing_rows].to(device),
                                     pool_ids[pair_rows].to(device)])
                    mask = torch.cat([mask[keep_t], pool_mask[sing_rows].to(device),
                                      pool_mask[pair_rows].to(device)])
                gold_keep = gold[keep_t]          # exact labels of retained seed rows (a2/a3 loss)
                n_seed_in_batch = len(keep)
                n_sing_in_batch = len(sing_rows)
            else:
                gold_keep = gold
                n_seed_in_batch = len(idx)
                n_sing_in_batch = 0

            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.time()
            else:
                t0 = None

            if global_step == 0:
                watchdog.start()

            opt.zero_grad()
            z = model(ids, mask)   # ONE forward on the (constant-size) batch — all arms

            if arm == "a2":
                n_pair_in_batch = len(ids) - n_seed_in_batch - n_sing_in_batch
                # --- batch-prior logit adjustment (gamma=1.0) on the exactly-supervised rows ---
                batch_counts = torch.zeros(len(CLASSES_), device=device)
                if n_seed_in_batch > 0:
                    batch_counts.scatter_add_(0, gold_keep, torch.ones(n_seed_in_batch, device=device))
                if n_sing_in_batch > 0:
                    batch_counts.scatter_add_(0, sing_labels.to(device), torch.ones(n_sing_in_batch, device=device))
                if n_pair_in_batch > 0:
                    batch_counts[i_rab] += 0.5 * n_pair_in_batch   # pair rows: 1/2 to RAB, 1/2 to TUN
                    batch_counts[i_tun] += 0.5 * n_pair_in_batch
                f_batch = batch_counts / max(1, len(ids))
                f_batch = f_batch.clamp_min(1.0 / max(1, len(ids)))   # floor for absent classes (log finite)
                log_f = torch.log(f_batch)
                n_exact = n_seed_in_batch + n_sing_in_batch
                z_exact = z[:n_exact] - GAMMA * log_f.unsqueeze(0)    # logit-adjusted exact rows
                ce_seed = ce_none(z_exact[:n_seed_in_batch], gold_keep)
                ce_sing = ce_none(z_exact[n_seed_in_batch:], sing_labels.to(device))
                if n_pair_in_batch > 0:
                    pi_tilde_rows = anchor["table"][anchor["pos2row"][pair_rows]].to(device)
                    pl_pair = partial_label_anch_loss(z[n_exact:], pi_tilde_rows, i_rab, i_tun,
                                                      LAMBDA_KL, LAMBDA_M)
                    loss = torch.cat([ce_seed, ce_sing, pl_pair]).mean()
                else:
                    loss = torch.cat([ce_seed, ce_sing]).mean()
                if LAMBDA_CONTRAST > 0.0 and n_seed_in_batch > 0:
                    z_seed = z[:n_seed_in_batch]
                    rab_m = (gold_keep == i_rab)
                    tun_m = (gold_keep == i_tun)
                    _ct = []
                    if bool(rab_m.any()):
                        _ct.append((CONTRAST_MARGIN - (z_seed[rab_m, i_rab] - z_seed[rab_m, i_tun])).clamp_min(0.0))
                    if bool(tun_m.any()):
                        _ct.append((CONTRAST_MARGIN - (z_seed[tun_m, i_tun] - z_seed[tun_m, i_rab])).clamp_min(0.0))
                    if _ct:
                        loss = loss + LAMBDA_CONTRAST * torch.cat(_ct).mean()
            elif arm == "a3":
                # twin: pool rows excluded from the loss; seed rows only.
                loss = nn.CrossEntropyLoss()(z[:n_seed_in_batch], gold_keep)
            else:
                loss = nn.CrossEntropyLoss()(z, gold)       # pure anchor

            # Engage probe (a2 only, frozen steps): in-batch pool-row grad-norm^2 share.
            if arm == "a2" and (global_step + 1) in ENGAGE_STEPS_:
                n_pool_in_batch = len(ids) - n_seed_in_batch
                if n_pool_in_batch > 0:
                    ce_seed_p = (ce_none(z[:n_seed_in_batch], gold_keep).mean()
                                 if n_seed_in_batch > 0 else torch.tensor(0.0, device=device))
                    n_pair_in_batch_e = len(ids) - n_seed_in_batch - n_sing_in_batch
                    ce_sing_p = ce_none(z[n_seed_in_batch:n_seed_in_batch + n_sing_in_batch], sing_labels.to(device)).mean()
                    if n_pair_in_batch_e > 0:
                        pi_tilde_rows_e = anchor["table"][anchor["pos2row"][pair_rows]].to(device)
                        pl_pair_e = partial_label_anch_loss(
                            z[n_seed_in_batch + n_sing_in_batch:], pi_tilde_rows_e, i_rab, i_tun,
                            LAMBDA_KL, LAMBDA_M).mean()
                        loss_pool_part = (ce_sing_p * n_sing_in_batch + pl_pair_e * n_pair_in_batch_e) / n_pool_in_batch
                    else:
                        loss_pool_part = ce_sing_p
                    w_s, w_p = n_seed_in_batch / len(ids), n_pool_in_batch / len(ids)
                    g_p = torch.autograd.grad(w_p * loss_pool_part, model.parameters(),
                                              retain_graph=True, allow_unused=True)
                    g_s = torch.autograd.grad(w_s * ce_seed_p, model.parameters(),
                                              retain_graph=True, allow_unused=True)
                    n2_p = sum(torch.norm(g) ** 2 for g in g_p if g is not None)
                    n2_s = sum(torch.norm(g) ** 2 for g in g_s if g is not None)
                    share = float(n2_p / (n2_p + n2_s)) if (n2_p + n2_s).item() > 0 else None
                    rec = {"step": global_step + 1, "event": "engage_probe",
                           "grad_norm2_pool": round(float(n2_p), 6),
                           "grad_norm2_seed": round(float(n2_s), 6),
                           "pool_share_pct": round(share * 100, 4) if share is not None else None,
                           "n_pool_rows": n_pool_in_batch, "n_seed_rows": n_seed_in_batch}
                    metrics_log.write(json.dumps(rec) + "\n")
                    metrics_log.flush()
                    print(f"[engage] {json.dumps(rec)}", flush=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step == 0:
                watchdog.check()

            if t0 is not None:
                torch.cuda.synchronize()
                gpu_busy_s += time.time() - t0

            n_repl_rows_total += (len(ids) - n_seed_in_batch) if arm in ("a2", "a3") else 0
            global_step += 1
            if global_step % 25 == 0 or global_step == total_steps or global_step == 1:
                rec = {"step": global_step, "epoch": epoch + 1, "loss": loss.item(),
                       "n_pool_rows_in_batch": int(len(ids) - n_seed_in_batch) if arm in ("a2", "a3") else 0}
                metrics_log.write(json.dumps(rec) + "\n")
                metrics_log.flush()
                write_step_event(tb_writer, global_step, rec)   # TB mirror of the JSONL stream
                print(f"[step {global_step}/{total_steps}] " +
                      " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()), flush=True)
        epoch += 1
        ev = evaluate(model, mon_ids, mon_mask, mon_labels, device, classes=CLASSES_, class2idx=CLASS2IDX_)
        rec = {"step": global_step, "epoch": epoch, "val2_macro_f1": ev["macro_f1"],
               "val2_accuracy": ev["accuracy"], "val2_loss": ev["loss"],
               "val2_pair_mass_pct": ev.get("pair_mass_pct"),
               "val2_per_class_f1": {c: v["f1"] for c, v in ev["per_class"].items()}}
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                             if isinstance(v, float)) +
              " per_class=" + json.dumps(rec["val2_per_class_f1"]), flush=True)

        # trailing-3-mean selection (e >= 3; ties within SEL_TIE_TOL -> latest epoch). INHERITED.
        val2_hist.append(ev["macro_f1"])
        if len(val2_hist) >= 3:
            score = sum(val2_hist[-3:]) / 3.0
            if score > best_score + SEL_TIE_TOL or (abs(score - best_score) <= SEL_TIE_TOL and global_step > best_step):
                best_score, best_step, best_f1 = score, global_step, ev["macro_f1"]
                save_checkpoint(str(out_dir / "best.pt"), model, opt, global_step, epoch,
                                objective={"best_score": best_score, "best_step": best_step,
                                           "best_f1": best_f1, "val2_hist": val2_hist,
                                           "val2_macro_f1": ev["macro_f1"]},
                                rng_state={"global": save_rng_state(),
                                           "gen_stream": gen_stream.get_state(),
                                           "gen_pool": gen_pool.get_state(),
                                           "gen_rem": gen_rem.get_state()},
                                config_sha256=config_sha256,
                                extra={"epoch_perm": perm, "config": cfg,
                                       "classes": CLASSES_, "backbone": cfg["backbone"]})
        # per-epoch latest checkpoint (crash-safe resume source; atomic + sha256)
        save_checkpoint(str(out_dir / "last.pt"), model, opt, global_step, epoch,
                        objective={"best_score": best_score, "best_step": best_step,
                                   "best_f1": best_f1, "val2_hist": val2_hist,
                                   "val2_macro_f1": ev["macro_f1"]},
                        rng_state={"global": save_rng_state(),
                                   "gen_stream": gen_stream.get_state(),
                                   "gen_pool": gen_pool.get_state(),
                                   "gen_rem": gen_rem.get_state()},
                        config_sha256=config_sha256,
                        extra={"epoch_perm": perm, "config": cfg,
                               "classes": CLASSES_, "backbone": cfg["backbone"]})
        write_step_event(tb_writer, global_step, rec)   # TB mirror of the JSONL stream

        if (EARLY_STOP_FLOOR_ is not None and global_step == EARLY_STOP_STEP_
                and ev["macro_f1"] < EARLY_STOP_FLOOR_):
            early_stopped = True
            stop = True
            print(f"[early-stop] {mon_split} {ev['macro_f1']:.4f} < {EARLY_STOP_FLOOR_} at step "
                  f"{global_step} — stopping, NO TEST read", flush=True)

    if args.smoke_eval:
        ev_smoke = evaluate(model, mon_ids, mon_mask, mon_labels, device, classes=CLASSES_, class2idx=CLASS2IDX_)
        rec = {"step": global_step, "event": "smoke_eval", "val2_macro_f1": ev_smoke["macro_f1"],
               "val2_per_class_f1": {c: v["f1"] for c, v in ev_smoke["per_class"].items()}}
        metrics_log.write(json.dumps(rec) + "\n")
        metrics_log.flush()
        print(f"[smoke-eval] {json.dumps(rec)}", flush=True)

    save_checkpoint(str(out_dir / "last.pt"), model, opt, global_step, epoch,
                    objective={"best_score": best_score, "best_step": best_step,
                               "best_f1": best_f1, "val2_hist": val2_hist},
                    rng_state={"global": save_rng_state(),
                               "gen_stream": gen_stream.get_state(),
                               "gen_pool": gen_pool.get_state(),
                               "gen_rem": gen_rem.get_state()},
                    config_sha256=config_sha256,
                    extra={"config": cfg, "classes": CLASSES_, "backbone": cfg["backbone"]})

    test_result = None
    if not early_stopped and not args.no_test:
        assert best_step > 0, "no trailing-3-mean checkpoint selected (run too short?)"
        eval_id = f"{CYCLE_MARK}-{run_name}"
        if dataset == "madar":
            test_views = build_views(cfg["seed_fraction"])["test"]
            test_texts = texts_for(test_views)
            test_labels = torch.tensor([CLASS2IDX_[r["label"]] for r in test_views], dtype=torch.long)
        else:
            test_views = views["test"]
            test_texts = [elhaj_text_index()[r] for r, _c in test_views]
            test_labels = torch.tensor([_c for _r, _c in test_views], dtype=torch.long)
        test_ids, test_mask = tokenize_bank(test_texts, tok, cfg["max_length"])
        best_ckpt = load_checkpoint(str(out_dir / "best.pt"), model, device=device)
        test_result = evaluate(model, test_ids, test_mask, test_labels, device, classes=CLASSES_, class2idx=CLASS2IDX_)
        json.dump({"eval_id": eval_id, "arm": arm, "checkpoint": "best.pt",
                   "checkpoint_step": best_step, "checkpoint_val2_macro_f1": best_f1,
                   "checkpoint_trailing3_score": best_score,
                   **{k: v for k, v in test_result.items()}},
                  open(out_dir / "test_report.json", "w"), indent=1)
        print(f"[test] {eval_id} macro_f1={test_result['macro_f1']:.4f} "
              f"per_class=" + json.dumps({c: v['f1'] for c, v in test_result['per_class'].items()}), flush=True)

    metrics_log.close()
    gpu_mon.stop()
    if tb_writer is not None:
        tb_writer.close()
    wandb_url = None
    if wb_run is not None:  # pragma: no cover - wandb optional
        try:
            wandb_url = wb_run.get_url()
            wb_run.finish()
        except Exception:
            pass

    wall_s = time.time() - t_start
    final = {"run_name": run_name, "arm": arm, "dataset": dataset,
             "seed_fraction": cfg["seed_fraction"], "seed": cfg["seed"],
             "config_sha256": config_sha256, "split_seed": SPLIT_SEED,
             "KS": KS, "KP": KP, "K": K,
             "total_steps": total_steps, "wall_seconds": round(wall_s, 1),
             "gpu_busy_seconds_step_time_basis": round(gpu_busy_s, 1),
             "gpu_hours_step_time_basis": round(gpu_busy_s / 3600.0, 4),
             "mean_step_time_s": round(gpu_busy_s / max(1, global_step), 6),
             "n_pool_row_uses": n_repl_rows_total,
             "best_trailing3_score": best_score, "best_step": best_step,
             "best_val2_macro_f1_at_best_step": best_f1,
             "early_stopped": early_stopped,
             "test_macro_f1": test_result["macro_f1"] if test_result else None,
             "device": str(device),
             "tb_logdir": str(log_dir / "tb") if tb_writer is not None else None,
             "wandb_url": wandb_url}
    write_summary_atomic(str(out_dir / "train_summary.json"), final)
    print(f"[done] {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
