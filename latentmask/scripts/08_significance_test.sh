#!/usr/bin/env bash
# ============================================================================
# 08_significance_test.sh — Bootstrap significance tests (Table 11)
# ============================================================================
set -euo pipefail

export nnUNet_results="${nnUNet_results:-$HOME/nnUNet_data/nnUNet_results}"

DATASET_ID="${1:-100}"
CONFIG="3d_fullres"

echo "=== Bootstrap significance tests ==="

python -c "
import json
from pathlib import Path
from latentmask.utils.metrics import paired_bootstrap_test

results_root = Path('${nnUNet_results}')

# Find dataset dir
dataset_dirs = list(results_root.glob('Dataset${DATASET_ID}_*'))
if not dataset_dirs:
    print('No results found')
    exit(1)
dataset_dir = dataset_dirs[0]

# Load per-case results for each method
def load_cv_metrics(trainer_name):
    path = dataset_dir / f'{trainer_name}__nnUNetPlans__${CONFIG}' / 'cv_metrics.json'
    if not path.exists():
        print(f'  WARNING: {path} not found')
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('per_case', [])

comparisons = [
    ('LatentMaskTrainer', 'LatentMaskTrainer_A1_UniformPU', 'LatentMask vs A1 (uniform PU)'),
    ('LatentMaskTrainer', 'nnPUSegTrainer', 'LatentMask vs nnPU-Seg'),
    ('LatentMaskTrainer', 'nnUNetTrainer', 'LatentMask vs nnUNet'),
]

print(f'{'Comparison':<45} {'Dice p-val':<15} {'F1 p-val':<15} {'Sig?':<5}')
print('-' * 80)

for trainer_a, trainer_b, name in comparisons:
    cases_a = load_cv_metrics(trainer_a)
    cases_b = load_cv_metrics(trainer_b)

    if cases_a is None or cases_b is None:
        print(f'{name:<45} N/A')
        continue

    # Match cases
    a_dict = {c['case_id']: c for c in cases_a}
    b_dict = {c['case_id']: c for c in cases_b}
    common = sorted(set(a_dict.keys()) & set(b_dict.keys()))

    if len(common) < 5:
        print(f'{name:<45} Too few common cases ({len(common)})')
        continue

    dice_a = [a_dict[k]['dice'] for k in common]
    dice_b = [b_dict[k]['dice'] for k in common]
    f1_a = [a_dict[k]['lesion_f1'] for k in common]
    f1_b = [b_dict[k]['lesion_f1'] for k in common]

    dice_test = paired_bootstrap_test(dice_a, dice_b)
    f1_test = paired_bootstrap_test(f1_a, f1_b)

    sig = '✓' if dice_test['p_value'] < 0.05 and f1_test['p_value'] < 0.05 else '✗'
    print(f'{name:<45} {dice_test[\"p_value\"]:<15.4f} {f1_test[\"p_value\"]:<15.4f} {sig:<5}')

print()
"

echo "=== Significance tests complete ==="
