#!/usr/bin/env bash
# Phase A1 — generate offline box annotations (LiTS).
#
# One invocation, fold=0, produces a fold-invariant benchmark:
#   data/box_annotations/P-uniform/     # per-scan JSONs + box_segmentations/
#   data/box_annotations/P-mild/
#   data/box_annotations/P-steep/
#
# Why fold=0 (not per-fold):
#   generate_box_annotations.py emits boxes for ALL 131 scans; only the
#   scale_factor (which makes E[retention] = target_R) depends on the
#   fold's train_keys' mu. We fix the scale_factor to fold-0's reference
#   so every fold sees IDENTICAL boxes — the benchmark is fold-invariant.
#   The method's fold-specific information lives in _calibration_fold{f}.pkl,
#   produced by a2_run_all_calibrations.sh. This cleanly separates
#   "benchmark" (fixed, shared) from "method calibration" (fold-specific).
#
# ~5 min CPU. Idempotent — re-run to refresh boxes after changing the
# simulator. Will refuse to overwrite existing box_segmentations unless
# FORCE=1.

set -euo pipefail

: "${nnUNet_preprocessed:?set nnUNet_preprocessed}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATASET_DIR="$nnUNet_preprocessed/Dataset501_LiTS"
OUT_ROOT="$REPO/data/box_annotations"
FG_LABEL=2
PIXEL_FRACTION=0.3
SEED=42
REF_FOLD=0   # scale-factor reference fold (see header)

if [[ -d "$OUT_ROOT/P-steep" ]] && compgen -G "$OUT_ROOT/P-steep/*.json" > /dev/null; then
  if [[ "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $OUT_ROOT/P-steep already populated."
    echo "       Set FORCE=1 to regenerate (this will invalidate existing calibrations!)."
    exit 0
  fi
  echo "[force] regenerating — existing _calibration_fold*.pkl will be stale."
fi

mkdir -p "$OUT_ROOT"
python -m latentmask.scripts.generate_box_annotations \
  --dataset_dir     "$DATASET_DIR" \
  --output_dir      "$OUT_ROOT" \
  --fold            "$REF_FOLD" \
  --pixel_fraction  "$PIXEL_FRACTION" \
  --target_R        0.70 \
  --fg_label        "$FG_LABEL" \
  --seed            "$SEED"

echo
echo "Done. Check actual retention rates:"
for P in P-uniform P-mild P-steep; do
  if [[ -f "$OUT_ROOT/$P/_summary.json" ]]; then
    printf "  %-10s " "$P"
    python3 -c "
import json
s = json.load(open('$OUT_ROOT/$P/_summary.json'))
print(f\"scans={s.get('n_scans')}, retained={s.get('n_retained')}/{s.get('total_ccs')}, actual_R={s.get('actual_R', 'n/a')}\")
"
  fi
done
echo
echo "Gate: every protocol's actual_R should be within ~0.02 of target 0.70."
echo "Next: bash scripts/a2_run_all_calibrations.sh"
