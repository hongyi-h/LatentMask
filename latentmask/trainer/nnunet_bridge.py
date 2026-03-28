"""
Bridge module: place this file inside nnunetv2/training/nnUNetTrainer/
so that nnUNet's recursive_find_python_class discovers LatentMask trainers.

Usage:
    cp latentmask/trainer/nnunet_bridge.py nnUNet/nnunetv2/training/nnUNetTrainer/latentmask_trainers.py

Then use:
    nnUNetv2_train DATASET CONFIG FOLD -tr LatentMaskTrainer
"""

# Re-export all trainers so nnUNet can find them
from latentmask.trainer.latentmask_trainer import (  # noqa: F401
    LatentMaskTrainer,
    LatentMaskTrainer_A1_UniformPU,
    LatentMaskTrainer_A2_PropNetOnly,
    LatentMaskTrainer_A3_NoRefine,
    MixedNaiveTrainer,
    nnPUSegTrainer,
    MeanTeacher3DTrainer,
    CPSTrainer,
    BoxSup3DTrainer,
)
