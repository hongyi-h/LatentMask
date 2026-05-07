#!/usr/bin/env bash
# Phase A2 — fit g_θ for every (protocol × fold) and pickle the calibration
# artifact that the trainer LOADs at on_train_start.
#
# Produces:
#   data/box_annotations/{P-uniform,P-mild,P-steep}/_calibration_fold{0..4}.pkl
#   results/m1_calibration_v6/calibration_{protocol}_v6.json    (human report)
#   results/m1_calibration_v6/calibration_{protocol}_fold{f}.pkl (audit shadow)
#
# 15 fits × ~2–5 min on CPU = ~45 min total. Idempotent — re-run overwrites.
# Requires Phase A1 boxes already on disk.

set -euo pipefail

: "${nnUNet_preprocessed:?set nnUNet_preprocessed}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATASET_DIR="$nnUNet_preprocessed/Dataset501_LiTS"
BOX_ROOT="$REPO/data/box_annotations"
OUT_DIR="$REPO/results/m1_calibration_v6"
FG_LABEL=2
PIXEL_FRACTION=0.3
SEED=42
ECE_GATE="${ECE_GATE:-0.15}"   # LiTS: 136 CCs → gate relaxed to 0.15

# Which folds. B1 core needs all 5. Ablations use fold 0/1 only — still cheaper
# to fit them all upfront than to come back later.
FOLDS=${FOLDS:-"0 1 2 3 4"}
PROTOCOLS=${PROTOCOLS:-"P-uniform P-mild P-steep"}

mkdir -p "$OUT_DIR"

# Sanity: Phase A1 ran
for P in $PROTOCOLS; do
  if ! compgen -G "$BOX_ROOT/$P/*.json" > /dev/null; then
    echo "ERROR: $BOX_ROOT/$P is empty. Run scripts/a1_generate_box_annotations.sh first."
    exit 1
  fi
done

FAIL=0
for P in $PROTOCOLS; do
  for F in $FOLDS; do
    echo "=== M1 fit: protocol=$P fold=$F ==="
    if ! python -m latentmask.scripts.run_calibration_cv \
        --dataset_dir           "$DATASET_DIR" \
        --box_annotations_dir   "$BOX_ROOT/$P" \
        --protocol              "$P" \
        --fold                  "$F" \
        --pixel_fraction        "$PIXEL_FRACTION" \
        --fg_label              "$FG_LABEL" \
        --seed                  "$SEED" \
        --ece_gate              "$ECE_GATE" \
        --output                "$OUT_DIR" ; then
      echo "FAIL: $P fold=$F"
      FAIL=1
    fi
  done
done

if (( FAIL )); then
  echo
  echo "At least one calibration fit failed. Not safe to launch M2."
  exit 1
fi

echo
echo "=== Gate summary (ece_gate = $ECE_GATE) ==="
for P in $PROTOCOLS; do
  for F in $FOLDS; do
    REP="$OUT_DIR/calibration_${P}_v6.json"
    ART="$BOX_ROOT/$P/_calibration_fold${F}.pkl"
    if [[ ! -f "$ART" ]]; then
      echo "  MISSING ARTIFACT: $ART"
      FAIL=1
    fi
  done
done

python3 - <<'PY'
import json, os, glob
out = 'results/m1_calibration_v6'
reps = sorted(glob.glob(os.path.join(out, 'calibration_*_fold*_v6.json')))
if not reps:
    print("  (no reports found — did the fits run?)")
for rep in reps:
    r = json.load(open(rep))
    print(f"  {r['protocol']:<10} fold={r.get('fold', '?')}  "
          f"cv_max_ece={r.get('cv_max_ece', float('nan')):.4f}  "
          f"gate_pass={r.get('gate_pass')}")
PY

echo
if (( FAIL )); then
  echo "One or more artifacts missing. Inspect logs above."
  exit 1
fi
echo "All calibrations produced. Next: sanity overfit (python scripts/run.py R003 --num_epochs 5)."
