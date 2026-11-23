# IDEP — Inverted-Decoupling Entropy-regularized Partial-label pool-row Mixing

Reproduction code for **IDEP**, a weakly supervised Arabic dialect identification
method evaluated on MADAR Corpus-6 (6 fine classes: MSA/BEI/CAI/DOH/RAB/TUN).
It replaces a fixed fraction `K=16` of each seed batch with exact-lifted
singleton-fiber pool rows (constant batch, ~1.0× step cost) using two components:

1. **INVERTED-DECOUPLING** — removal weights `ρ ∝ seed F1` (strongest class
   MSA removed most, weakest DOH least) with injection weights `π ∝ 1/seed F1`
   (DOH-heaviest), plus a batch-prior logit adjustment
   `z'_c = z_c − γ·log f_c^batch`, `γ=1.0`, on the 28 exactly-supervised rows.
2. **ENTROPY-REGULARIZED PARTIAL-LABEL** — the Maghreb pair-row loss is the
   admissible-set marginal `−log Σ_{c∈S} p_c` plus an intra-fiber entropy
   penalty `β·H(q)`, `β=0.5`, on `{RAB,TUN}` (first direct RAB↔TUN
   disambiguation in the lineage).

Backbone weights are `UBC-NLP/MARBERT` (Hugging Face) or as set per config.
No patient data is included; datasets are not redistributed.

## Setup

```bash
uv sync   # Python 3.12; torch 2.11.0+cu130 from the PyTorch cu130 index
```

The stack is pinned in `pyproject.toml` and `uv.lock`. `transformers==4.40.2`
requires `huggingface-hub==0.34.4` (pinned).

## Data

Set `IDEP_DATA_ROOT` (default `./data`) to a directory containing:

```
$IDEP_DATA_ROOT/madar_shared_task_2019/MADAR-Shared-Task-Subtask-1/
$IDEP_DATA_ROOT/elhaj_arabic_dialects_2018/
```

`splits/` ships the frozen group-level splits. These files store no sentence
text, so the datasets must be present. The split seed `2026-11-20-RUN-2026-11-20-02`
is part of the data identity; do not change it.

## Run

```bash
export IDEP_DATA_ROOT=/path/to/data

# IDEP (GPU required; --allow-cpu overrides). Final TEST runs unless --no-test.
uv run python idemp/train_idemp.py --config idemp/configs/final_v12_a2_s42.json

# Baseline T2 (MARBERT, plain)
uv run python baselines/train_baseline.py --config baselines/configs/t2_plain_seed10.json

# Locked TEST evaluation of a baseline checkpoint
uv run python baselines/eval_final.py --config baselines/configs/<config>.json --split test --eval-id <id>
```

`--steps-cap N` limits steps for smoke runs. The anchor table is shipped as
`splits/anchor_table.pt` with a `.sha256` sidecar; rebuild it with
`uv run python idemp/build_anchor_table.py`.

## External dependency (baseline B3)

B3 (hierarchical) needs an external checkout of `hierarchical_arabic_dialect_id`
and KenLM language-model binaries:

- `HIER_ARABIC_DIR`: checkout root (provides `classification/` and `data_preprocess/`).
- `B3_LM_ROOT`: directory with the KenLM binaries.

## Results

- `results/idemp/<run>/`: `train_summary.json`, `test_report.json`.
- `results/baselines/<run>/`: `test_report.json`.

IDEP arm `a2`, MADAR Corpus-6 TEST macro-F1, seeds 42/43/44: 87.86, 87.91, 87.90
(mean 87.89), from `results/idemp/final_v12_a2_s{42,43,44}/test_report.json`.

## Verification (CPU-only)

```bash
uv run python -m compileall -q common idemp baselines data_prep
```

## Arms

- `a0`: pure anchor (no replacement); step-time reference.
- `a2`: IDEP full (replacement on; logit-adjusted exact CE + entropy-regularized partial-label marginal).
- `a3`: anchor-removed twin (same cadence and draws; pool rows forwarded for compute matching, excluded from loss); isolates the pool signal.

## Determinism and selection

`gen_stream = Generator(seed)` (shuffle, anchor parity), `gen_pool = Generator(seed+1)`
(injection), `gen_rem = Generator(seed+2)` (removal). Draws are arm-independent, so
a2 and a3 see identical batches. Selection is the trailing-3-mean validation macro-F1
(ties within 1e-4 go to the later checkpoint). Early stop: if validation macro-F1 at
step 760 is below 0.8712, training stops and TEST is not read. TEST is read once per arm.
