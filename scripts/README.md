# Experiment runbook (v6.1)

**Single source of truth**: every runnable experiment has exactly one row in
[`runs.csv`](./runs.csv), keyed by the same IDs used in
`refine-logs/EXPERIMENT_TRACKER.md`. The launcher in
[`run.py`](./run.py) reads that row and dispatches — don't hand-write new
shell files for each run.

If you find yourself about to write `run_c4_f3.sh`, stop and add a row to
`runs.csv` instead.

## Why this shape

- **48 GPU runs, 15 calibrations.** One bash script per run = ~60 near-duplicate
  files that drift out of sync with EXPERIMENT_TRACKER. Already lived through
  that with v5 — never again.
- **CSV + launcher** gives:
  - `grep R027 runs.csv` = reproduce any paper number
  - `python run.py R027 --dry-run` = see the exact command before spending GPU-h
  - `python run.py --all --status` = see what's done vs pending
  - Status written back to CSV → survives session death

## Workflow

Everything below runs on the GPU server (needs nnunetv2 + CUDA + preprocessed
LiTS at `$nnUNet_preprocessed/Dataset501_LiTS`).

```bash
cd $LM_ROOT      # /Users/bugmakerh/BugMakerH/Work/QUT/LatentMask or equivalent
export nnUNet_raw=...
export nnUNet_preprocessed=...
export nnUNet_results=...
```

### Phase A — prerequisites (one time, ~2h CPU)

```bash
# A1. Generate box annotations for all 3 protocols (produces the box JSONs +
#     box_segmentations/ that both calibration and training will read).
bash scripts/a1_generate_box_annotations.sh

# A2. Fit g_θ for every (protocol × fold) and produce the pickled calibration
#     artifact (15 × ~5min on CPU). Reads output of A1, writes
#     _calibration_fold{f}.pkl into each protocol dir.
bash scripts/a2_run_all_calibrations.sh

# A3. Sanity: 1 scan, 5 epochs, C4 on P-steep f0. BLOCKING gate for M2.
#     Must see loss decrease + coverage_ratio > 0.05 + n_boundary/n_ccs < 0.8.
#     Notes on what to check: notes/sanity_check_rubric.md
python run.py R003 --override LM_NUM_EPOCHS=5 --override LM_NUM_ITERATIONS_PER_EPOCH=5
```

### Phase B — main experiments

```bash
python run.py --block B1    # R007–R031 (25 runs, 5-fold CV core)
python run.py --block B1c   # R032/R032b/R033/R033b (C2.5 + C4-inv, 2 folds each)
python run.py --block B1b   # R034–R041 (multi-protocol robustness)
python run.py --block B1d   # R042 (transfer)
python run.py --block B4    # R043–R048 (sensitivity)
python run.py --block B5    # R049 (qualitative)
python run.py --block B6    # R050–R052 (BraTS-METS, blocked on BraTS prep)
```

Alternatively run one-at-a-time:

```bash
python run.py R027 --seed 42
python run.py R027 --seed 1 --output_suffix seed1     # reproducibility ablation
```

### Phase C — aggregate

```bash
python scripts/c1_aggregate.py > notes/results_aggregate_$(date +%Y-%m-%d).md
```
(Not yet implemented — write it only once there's real data.)

## File map

```
scripts/
├── README.md                       ← this file
├── runs.csv                        ← 1 row = 1 experiment. The truth.
├── run.py                          ← dispatcher. Reads runs.csv → launches.
├── a1_generate_box_annotations.sh  ← protocols × folds (CPU)
├── a2_run_all_calibrations.sh      ← g_θ fit for every (proto, fold)
└── logs/                           ← stdout/stderr per run (git-ignored)
```

Status of each run lives in the `status` column of `runs.csv`. Possible
values: `TODO`, `RUNNING`, `DONE`, `FAILED`, `SKIP`. Update via
`python run.py R027 --mark DONE`.
