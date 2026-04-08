"""Bridge file for nnUNet trainer discovery.

Copy this file to nnUNet/nnunetv2/training/nnUNetTrainer/ to enable
CLI usage: nnUNetv2_train 501 3d_fullres 0 -tr LatentMaskTrainer

This is only needed for nnUNet CLI compatibility.
The launch_training.py script does NOT need this file.
"""
from latentmask.trainer.latentmask_trainer import LatentMaskTrainer  # noqa: F401
