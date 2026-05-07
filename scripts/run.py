#!/usr/bin/env python3
"""Experiment dispatcher for LatentMask v6.1.

Single source of truth: `scripts/runs.csv`. Each row is one experiment.
This launcher reads a row (by run_id or by --block) and turns it into a
`python -m latentmask.scripts.launch_training` invocation, with the right
env vars and arguments. Dispatches via nothing fancy — subprocess.run.

Usage:
    python run.py R027                         # one run
    python run.py --block B1                   # all B1 runs (TODO only)
    python run.py R003 --override LM_NUM_EPOCHS=5
    python run.py R027 --dry-run               # print command, don't execute
    python run.py R027 --mark DONE             # update status, no launch
    python run.py --status                     # show counts per status
    python run.py --list B1                    # list all B1 rows

C0 (upper bound) special case: C0 runs use neg_mode=none BUT
pixel_fraction=1.0 so every training scan contributes pixel GT. That's
triggered by variant=C0_upperbound.

B1d transfer (R042) special case: train on P-mild boxes, but use the
P-steep-fitted g_θ. We copy the P-steep fold-0 calibration artifact into
a shadow dir and point LM_BOX_ANNOTATIONS_DIR at P-mild boxes + that
shadow artifact.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS_CSV = HERE / 'runs.csv'
LOG_DIR = HERE / 'logs'
BOX_DIR_ROOT = REPO / 'data' / 'box_annotations'
DATASET_DIR_DEFAULT = {
    'LiTS': 'Dataset501_LiTS',
    'BraTSMETS': 'Dataset502_BraTSMETS',
}


@dataclass
class Run:
    run_id: str
    block: str
    config: str        # neg_mode OR 'calib' / 'box_gen' / 'qualitative'
    protocol: str
    fold: str          # may be 'all'
    seed: int
    dataset: str
    fg_label: int
    variant: str
    status: str
    notes: str
    _line: int         # 1-indexed line in runs.csv for status update


def read_rows() -> tuple[list[str], list[Run]]:
    """Return (raw_lines, parsed_rows). raw_lines retains comments + header."""
    raw = RUNS_CSV.read_text().splitlines()
    rows: list[Run] = []
    header: Optional[list[str]] = None
    for i, line in enumerate(raw):
        if not line.strip():
            continue
        if line.startswith('#'):
            continue
        fields = next(csv.reader([line]))
        if header is None:
            header = fields
            continue
        d = dict(zip(header, fields))
        rows.append(Run(
            run_id=d['run_id'], block=d['block'], config=d['config'],
            protocol=d['protocol'], fold=d['fold'],
            seed=int(d.get('seed') or 42),
            dataset=d['dataset'], fg_label=int(d['fg_label']),
            variant=d.get('variant', ''), status=d['status'],
            notes=d.get('notes', ''), _line=i))
    return raw, rows


def update_status(run_id: str, new_status: str) -> None:
    """Rewrite the status cell of one row in-place, preserving everything else."""
    import io
    raw = RUNS_CSV.read_text().splitlines()
    header: Optional[list[str]] = None
    for line in raw:
        if not line.strip() or line.startswith('#'):
            continue
        header = next(csv.reader([line]))
        break
    if header is None:
        sys.exit("ERROR: runs.csv has no header")
    status_idx = header.index('status')

    rewrote = False
    for idx, line in enumerate(raw):
        if not line.strip() or line.startswith('#'):
            continue
        fields = next(csv.reader([line]))
        if len(fields) <= status_idx:
            continue
        if fields[0] == 'run_id':
            continue  # header
        if fields[0] != run_id:
            continue
        prev = fields[status_idx]
        fields[status_idx] = new_status
        out = io.StringIO()
        csv.writer(out).writerow(fields)
        raw[idx] = out.getvalue().rstrip('\r\n')
        print(f"  {run_id}: {prev} -> {new_status}")
        rewrote = True
        break

    if not rewrote:
        sys.exit(f"ERROR: unknown run_id {run_id!r}")
    RUNS_CSV.write_text('\n'.join(raw) + '\n')


def box_dir_for(run: Run) -> Path:
    if run.dataset == 'LiTS':
        return BOX_DIR_ROOT / run.protocol
    if run.dataset == 'BraTSMETS':
        return REPO / 'data' / 'box_annotations_brats' / run.protocol
    raise ValueError(f"unknown dataset {run.dataset}")


def dataset_name_for(run: Run) -> str:
    return DATASET_DIR_DEFAULT[run.dataset]


def build_command(run: Run, overrides: dict[str, str],
                  num_epochs: Optional[int] = None,
                  output_suffix: Optional[str] = None,
                  dry_run: bool = False) -> tuple[list[str], dict[str, str]]:
    """Return (argv, extra_env) for launching one run."""
    # Variant-driven special cases
    pixel_fraction = 0.3
    if run.variant == 'C0_upperbound':
        pixel_fraction = 1.0
        neg_mode = 'none'
    else:
        neg_mode = run.config

    box_dir = box_dir_for(run)

    argv = [
        sys.executable, '-m', 'latentmask.scripts.launch_training',
        '--dataset_name', dataset_name_for(run),
        '--fold', run.fold,
        '--seed', str(run.seed),
        '--neg_mode', neg_mode,
        '--fg_label', str(run.fg_label),
        '--pixel_fraction', str(pixel_fraction),
        '--box_annotations_dir', str(box_dir),
    ]

    env: dict[str, str] = {}

    # Variant-driven env overrides
    v = run.variant
    if v == 'd_safe=3' or v == 'C2_d_safe=3':
        env['LM_D_MARGIN'] = '3'
    elif v == 'tau_low=0.2':
        env['LM_TAU_LOW'] = '0.2'
    elif v == 'tau_low=0.4':
        env['LM_TAU_LOW'] = '0.4'
    elif v == 'alpha_min=0.02':
        env['LM_ALPHA_MIN'] = '0.02'
    elif v == 'alpha_min=0.10':
        env['LM_ALPHA_MIN'] = '0.10'
    elif v == 'C4_xfer_steep_to_mild':
        # For R042: use P-mild boxes for training, but g_θ fitted on P-steep.
        # Stage a shadow box dir that symlinks P-mild JSONs + box_segmentations
        # but substitutes the calibration artifact from P-steep (same fold).
        shadow = BOX_DIR_ROOT / f'{run.protocol}_xfer_from_P-steep'
        if not dry_run:
            _stage_transfer_shadow(src_boxes=box_dir,
                                    calib_from=BOX_DIR_ROOT / 'P-steep',
                                    dest=shadow,
                                    fold=int(run.fold))
        idx = argv.index('--box_annotations_dir')
        argv[idx + 1] = str(shadow)

    if num_epochs is not None:
        env['LM_NUM_EPOCHS'] = str(num_epochs)

    # User-provided overrides last
    for k, v_ in overrides.items():
        env[k] = v_

    return argv, env


def _stage_transfer_shadow(src_boxes: Path, calib_from: Path,
                            dest: Path, fold: int) -> None:
    """Build a shadow box_annotations dir that uses src_boxes' JSONs + seg,
    but the calibration artifact from calib_from."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'box_segmentations').mkdir(exist_ok=True)

    # Symlink all JSONs + box_segmentations from src
    for f in src_boxes.glob('*.json'):
        link = dest / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    src_seg = src_boxes / 'box_segmentations'
    if src_seg.is_dir():
        for f in src_seg.iterdir():
            link = dest / 'box_segmentations' / f.name
            if not link.exists():
                link.symlink_to(f.resolve())

    # Copy calibration artifact (not symlink — we may want to edit/
    # inspect without affecting the original)
    src_art = calib_from / f'_calibration_fold{fold}.pkl'
    if not src_art.is_file():
        raise SystemExit(
            f"Transfer run needs source calibration at {src_art}, "
            f"but it's missing. Run M1 on P-steep first.")
    dst_art = dest / f'_calibration_fold{fold}.pkl'
    shutil.copy2(src_art, dst_art)


def launch_one(run: Run, *, dry_run: bool, overrides: dict[str, str],
                num_epochs: Optional[int], output_suffix: Optional[str]) -> int:
    if run.config in ('calib', 'box_gen', 'qualitative'):
        print(f"  {run.run_id}: {run.config} is not dispatched via run.py.")
        if run.config == 'calib':
            print("    Run: bash scripts/a2_run_all_calibrations.sh")
        elif run.config == 'box_gen':
            print("    Run: bash scripts/a1_generate_box_annotations.sh")
        else:
            print("    Qualitative rendering is manual; see notes/.")
        return 0

    argv, env = build_command(run, overrides, num_epochs, output_suffix,
                               dry_run=dry_run)
    env_line = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
    cmd_line = ' '.join(shlex.quote(a) for a in argv)
    print(f"\n=== {run.run_id} ({run.variant or run.config}) ===")
    print(f"  {env_line}  {cmd_line}")

    if dry_run:
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'{run.run_id}_{stamp}.log'

    update_status(run.run_id, 'RUNNING')
    merged_env = {**os.environ, **env}
    with open(log_path, 'w') as f:
        f.write(f'# {run.run_id}\n# {env_line}  {cmd_line}\n\n')
        f.flush()
        proc = subprocess.run(argv, env=merged_env, stdout=f,
                              stderr=subprocess.STDOUT)
    new_status = 'DONE' if proc.returncode == 0 else 'FAILED'
    update_status(run.run_id, new_status)
    print(f"  -> {new_status}. Log: {log_path}")
    return proc.returncode


def cmd_status(rows: list[Run]) -> None:
    from collections import Counter
    by_status = Counter(r.status for r in rows)
    by_block_status: dict[str, Counter] = {}
    for r in rows:
        by_block_status.setdefault(r.block, Counter())[r.status] += 1
    print('Overall:', dict(by_status))
    print()
    print(f"{'block':<6}  {'TODO':>4}  {'RUNNING':>7}  {'DONE':>4}  {'FAILED':>6}  {'SKIP':>4}")
    for block in sorted(by_block_status):
        c = by_block_status[block]
        print(f"  {block:<4}  {c.get('TODO',0):>4}  {c.get('RUNNING',0):>7}  "
              f"{c.get('DONE',0):>4}  {c.get('FAILED',0):>6}  {c.get('SKIP',0):>4}")


def cmd_list(rows: list[Run], block_filter: Optional[str]) -> None:
    print(f"{'id':<6}  {'block':<4}  {'cfg':<10}  {'proto':<10}  "
          f"{'fold':<4}  {'variant':<26}  status")
    for r in rows:
        if block_filter and r.block != block_filter:
            continue
        print(f"  {r.run_id:<4}  {r.block:<4}  {r.config:<10}  "
              f"{r.protocol:<10}  {r.fold:<4}  {r.variant:<26}  {r.status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_ids', nargs='*', help='Run IDs (e.g. R027). Omit with --block/--status/--list.')
    p.add_argument('--block', help='Run every TODO in this block (B0/B1/B1b/B1c/B1d/B3/B4/B5/B6)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--override', action='append', default=[],
                    metavar='KEY=VALUE',
                    help='Env var override, repeatable. E.g. --override LM_NUM_EPOCHS=5')
    p.add_argument('--seed', type=int, help='Override the seed column')
    p.add_argument('--num_epochs', type=int, help='Shortcut for --override LM_NUM_EPOCHS=N')
    p.add_argument('--output_suffix', help='Append to output folder tag')
    p.add_argument('--mark', choices=['TODO', 'RUNNING', 'DONE', 'FAILED', 'SKIP'],
                    help='Update status without launching')
    p.add_argument('--status', action='store_true', help='Print per-block status summary')
    p.add_argument('--list', dest='do_list', nargs='?', const='_ALL', default=None,
                    help='List runs (optionally filter by block)')
    p.add_argument('--rerun-failed', action='store_true',
                    help='When using --block, also rerun FAILED rows')
    p.add_argument('--force', action='store_true',
                    help='Also launch rows whose status is DONE')
    args = p.parse_args()

    _, rows = read_rows()

    if args.status:
        cmd_status(rows); return

    if args.do_list is not None:
        cmd_list(rows, None if args.do_list == '_ALL' else args.do_list)
        return

    if args.mark:
        if not args.run_ids:
            sys.exit("--mark requires at least one run_id")
        for rid in args.run_ids:
            update_status(rid, args.mark)
        return

    overrides = dict(kv.split('=', 1) for kv in args.override)

    # Resolve target list
    targets: list[Run]
    if args.block:
        def keep(r: Run) -> bool:
            if r.block != args.block:
                return False
            if r.status == 'DONE' and not args.force:
                return False
            if r.status == 'FAILED' and not args.rerun_failed:
                return False
            if r.status in ('RUNNING', 'SKIP'):
                return False
            return True
        targets = [r for r in rows if keep(r)]
        if not targets:
            print(f"No runs to launch in block {args.block} "
                  f"(use --force or --rerun-failed if you want DONE/FAILED too)")
            return
    else:
        if not args.run_ids:
            sys.exit("Provide at least one run_id or --block / --status / --list.")
        by_id = {r.run_id: r for r in rows}
        targets = []
        for rid in args.run_ids:
            if rid not in by_id:
                sys.exit(f"unknown run_id {rid!r}")
            targets.append(by_id[rid])

    if args.seed is not None:
        for t in targets:
            t.seed = args.seed

    ret_any_fail = 0
    for t in targets:
        rc = launch_one(t, dry_run=args.dry_run,
                         overrides=overrides,
                         num_epochs=args.num_epochs,
                         output_suffix=args.output_suffix)
        if rc != 0:
            ret_any_fail = rc
            # If running a whole block, keep going — most runs are independent.
    sys.exit(ret_any_fail)


if __name__ == '__main__':
    main()
