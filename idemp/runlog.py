"""run logging for the IDEP runner.

Observable record per run (all under the run's log dir):
  - console: structured loguru logs (every step/epoch + final metrics);
  - <log_dir>/metrics.jsonl : canonical attributed record (crash-safe append;
    preserved as the source of truth — the screen-era format is unchanged);
  - <log_dir>/tb/           : TensorBoard scalars on the SAME metrics stream
    (per-step loss + per-epoch monitoring metrics); direct run artifact path
    recorded in the final summary;
  - W&B (only when WANDB_MODE is set to online/offline by the launcher —
    never forced on); the run URL is logged and written to the summary when
    available;
  - <out_dir>/train_summary.json : final experiment record, written ATOMICALLY
    with mandatory-field validation (a record missing a mandatory field fails
    fast, never silently).
"""
from __future__ import annotations

import json
import os

try:
    from loguru import logger as _loguru
except ImportError:  # pragma: no cover - loguru is in the run environment
    import logging
    _loguru = logging.getLogger("idemp.runlog")

# mandatory fields of the final summary record
REQUIRED_SUMMARY_FIELDS = (
    "run_name", "arm", "dataset", "seed_fraction", "KS", "KP", "K",
    "seed", "config_sha256", "split_seed",
    "total_steps", "wall_seconds", "gpu_busy_seconds_step_time_basis",
    "gpu_hours_step_time_basis", "mean_step_time_s",
    "n_pool_row_uses", "best_trailing3_score", "best_step",
    "early_stopped", "test_macro_f1", "device",
)


class RunLogError(RuntimeError):
    pass


def init_runlog(log_dir: str, run_name: str):
    """loguru file sink + TensorBoard writer (+ W&B if configured).

    Returns (tb_writer, wandb_run_or_None). The JSONL stream stays canonical;
    TB/W&B are mirrors of the same metrics.
    """
    os.makedirs(log_dir, exist_ok=True)
    _loguru.add(os.path.join(log_dir, f"{run_name}.log"), level="INFO",
                format="{time:YYYY-MM-DDTHH:mm:ssZ} | {level} | {message}",
                enqueue=False)

    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(log_dir, "tb")
        tb_writer = SummaryWriter(log_dir=tb_dir)
        _loguru.info("tensorboard logdir: {}", tb_dir)
    except ImportError:  # pragma: no cover
        _loguru.warning("tensorboard unavailable; scalar events disabled")

    wb_run = None
    if os.environ.get("WANDB_MODE") in ("online", "offline"):
        try:
            import wandb
            wb_run = wandb.init(project="idemp", name=run_name, dir=log_dir,
                                mode=os.environ["WANDB_MODE"])
            url = getattr(wb_run, "get_url", lambda: "")()
            _loguru.info("wandb run: {} ({})", run_name, url)
        except Exception as e:  # pragma: no cover - wandb optional
            _loguru.warning("wandb init failed ({}); continuing without it", e)
            wb_run = None
    return tb_writer, wb_run


def write_step_event(tb_writer, step: int, rec: dict) -> None:
    """Mirror a metrics.jsonl step record to TB scalars (same stream)."""
    if tb_writer is None:
        return
    if "loss" in rec:
        tb_writer.add_scalar("train/loss", rec["loss"], step)
    if "n_pool_rows_in_batch" in rec:
        tb_writer.add_scalar("train/n_pool_rows_in_batch",
                             rec["n_pool_rows_in_batch"], step)
    for k, v in (rec.get("val2_per_class_f1") or {}).items():
        tb_writer.add_scalar(f"val2/f1_{k}", v, step)
    if "val2_macro_f1" in rec:
        tb_writer.add_scalar("val2/macro_f1", rec["val2_macro_f1"], step)
        tb_writer.add_scalar("val2/loss", rec.get("val2_loss", float("nan")), step)
        tb_writer.add_scalar("val2/accuracy",
                             rec.get("val2_accuracy", float("nan")), step)
    tb_writer.flush()


def validate_summary(hist: dict) -> list:
    """Missing mandatory fields (empty list = complete)."""
    return [k for k in REQUIRED_SUMMARY_FIELDS if k not in hist]


def write_summary_atomic(path: str, hist: dict) -> None:
    missing = validate_summary(hist)
    if missing:
        raise RunLogError(f"summary record missing mandatory fields: {missing}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hist, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
