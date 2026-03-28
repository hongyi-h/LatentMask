"""
Exponential Moving Average (EMA) teacher for LatentMask Stage 3 refinement.

Maintains a shadow copy of the student network weights, updated via:
    θ_teacher = α * θ_teacher + (1 - α) * θ_student
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMATeacher:
    """EMA shadow of a student network for pseudo-label generation."""

    def __init__(self, student: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.teacher = deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        """Update teacher weights from student."""
        for t_param, s_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            t_param.data.mul_(self.decay).add_(s_param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate pseudo-labels with teacher."""
        self.teacher.eval()
        return self.teacher(x)

    def state_dict(self) -> dict:
        return {
            "teacher_state": self.teacher.state_dict(),
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict):
        self.teacher.load_state_dict(state_dict["teacher_state"])
        self.decay = state_dict["decay"]

    def to(self, device):
        self.teacher.to(device)
        return self
