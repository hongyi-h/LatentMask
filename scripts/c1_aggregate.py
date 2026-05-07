#!/usr/bin/env python3
"""Aggregate LatentMask v6.1 experiment results into a single markdown table.

Reads:
  - scripts/runs.csv (status + metadata)
  - $nnUNet_results/Dataset*_*/{trainer}/{config_tag}/latentmask_diagnostics.json
  - $nnUNet_results/Dataset*_*/{trainer}/{config_tag}/evaluation/*.json (post-hoc)

Writes markdown to stdout. Intended usage:
    python scripts/c1_aggregate.py > notes/results_aggregate_$(date +%Y-%m-%d).md

This is INTENTIONALLY under-engineered. Rewrite once real evaluation outputs
exist — the eval JSON schema is not finalized yet.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_CSV = HERE / 'runs.csv'

# Columns to emit. Add metrics as the eval pipeline lands.
COLUMNS = [
    'run_id', 'block', 'config', 'protocol', 'fold', 'status',
    'dice_all', 'small_lesion_det_rate', 'fp_per_scan',
    'cv_max_ece_loaded', 'n_boundary_frac', 'coverage_ratio',
]


def read_runs():
    rows = []
    with open(RUNS_CSV) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def locate_output(run, nnunet_results: Path) -> Path | None:
    """Find the trainer output folder for this run, if it exists.

    Pattern (from launch_training.py):
        {nnunet_results}/Dataset{id}_{name}/nnUNetTrainer__nnUNetPlans__3d_fullres/
        fold_{fold}/{neg_mode}_{proto}_fold{fold}_seed{seed}/
    """
    dataset_map = {'LiTS': 'Dataset501_LiTS', 'BraTSMETS': 'Dataset502_BraTSMETS'}
    ds = dataset_map.get(run['dataset'])
    if ds is None:
        return None
    # Don't hard-code trainer/planner; glob for the config_tag folder instead
    neg_mode = 'none' if run.get('variant') == 'C0_upperbound' else run['config']
    proto = run['protocol']
    fold = run['fold']
    seed = run['seed']
    tag = f"{neg_mode}_{proto}_fold{fold}_seed{seed}"
    candidates = list(nnunet_results.glob(f'{ds}/*/*/{tag}'))
    return candidates[0] if candidates else None


def extract_metrics(out_dir: Path) -> dict:
    metrics = {}
    diag = out_dir / 'latentmask_diagnostics.json'
    calib_audit = out_dir / 'calibration_loaded.json'
    eval_json = out_dir / 'evaluation' / 'summary.json'

    if diag.is_file():
        d_all = json.loads(diag.read_text())
        # Diagnostics is a list of per-checkpoint snapshots; use the last.
        last = d_all[-1] if isinstance(d_all, list) and d_all else {}
        metrics['coverage_ratio'] = last.get('coverage_ratio_mean')
        n_b = last.get('n_boundary', 0)
        n_ccs = max(last.get('n_ccs_total', 1), 1)
        metrics['n_boundary_frac'] = n_b / n_ccs if n_ccs else None

    if calib_audit.is_file():
        c = json.loads(calib_audit.read_text())
        metrics['cv_max_ece_loaded'] = c.get('cv_max_ece')

    if eval_json.is_file():
        e = json.loads(eval_json.read_text())
        metrics['dice_all'] = e.get('dice_all')
        metrics['small_lesion_det_rate'] = e.get('small_lesion_det_rate')
        metrics['fp_per_scan'] = e.get('fp_per_scan')

    return metrics


def fmt(v) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def main():
    nnunet_results = Path(os.environ.get('nnUNet_results', 'nnUNet_results'))
    runs = read_runs()

    print(f'# Aggregate results ({os.environ.get("USER", "?")}, '
          f'{os.popen("date +%Y-%m-%d").read().strip()})\n')
    print(f'Source: `scripts/runs.csv` + `$nnUNet_results`')
    print(f'nnUNet_results = `{nnunet_results}`\n')

    print('| ' + ' | '.join(COLUMNS) + ' |')
    print('|' + '|'.join(['---'] * len(COLUMNS)) + '|')

    for r in runs:
        row = {k: r.get(k, '') for k in COLUMNS}
        if r['status'] == 'DONE':
            out = locate_output(r, nnunet_results)
            if out is not None:
                row.update(extract_metrics(out))
        print('| ' + ' | '.join(fmt(row.get(c, '')) for c in COLUMNS) + ' |')

    print()
    print(f'Rows with DONE status: '
          f'{sum(1 for r in runs if r["status"] == "DONE")} / {len(runs)}')


if __name__ == '__main__':
    main()
