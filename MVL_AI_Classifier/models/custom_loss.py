"""
Multi-view loss combining supervised classification on a fusion head and multiple branches, plus cross-branch knowledge distillation.

It sums:
1) Fusion cross-entropy loss (main prediction),
2) Branch cross-entropy losses (auxiliary supervision),
3) Pairwise KL-based distillation between branches to enforce consistency

# Teacher logits are raw prediction scores from a larger, pre-trained model (the "expert"),
# while student logits are raw scores from the smaller model being trained to mimic the teacher.
# Both are unnormalized outputs (before softmax) and are compared during knowledge distillation.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss using KL divergence.

    This loss encourages the student model to match the softened
    probability distribution of the teacher model.

    Args:
        temperature (float): Controls softness of probability distribution.
            Higher values produce smoother distributions.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()

        self.temperature = temperature

        # KL divergence loss expects log-probabilities as input
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher distributions.

        Args:
            student_logits (Tensor): Raw logits from student model.
            teacher_logits (Tensor): Raw logits from teacher model.

        Returns:
            Tensor: Scalar distillation loss.
        """

        t = self.temperature

        # Apply temperature (Temperature controls how soft or sharp the model’s predicted probabilities are) scaling:
        # Student uses log_softmax
        # Teacher uses softmax to form target probability distribution
        student_log_probs = F.log_softmax(student_logits / t, dim=1)
        teacher_probs = F.softmax(teacher_logits / t, dim=1)

        # KL divergence between teacher (target) and student (input)
        return self.kl(student_log_probs, teacher_probs)


class MultiViewLoss(nn.Module):
    """
    Multi-view learning loss combining:
    1. Fusion classification loss
    2. Branch-wise classification loss
    3. Cross-branch knowledge distillation loss

    This is useful in multi-branch architectures where:
    - Each branch learns complementary representations
    - A fusion head produces final predictions
    - Branches are encouraged to align via distillation
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, temperature: float = 2.0):
        """
        Args:
            alpha (float): Weight for branch classification loss.
            beta (float): Weight for distillation loss between branches.
            temperature (float): Temperature for KD smoothing.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # Standard classification loss for supervised learning
        self.ce = nn.CrossEntropyLoss()

        # Distillation loss for aligning branch outputs
        self.kd = DistillationLoss(temperature)

    def classification_loss(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy classification loss.

        Args:
            preds (Tensor): Model logits.
            target (Tensor): Ground-truth labels.

        Returns:
            Tensor: CE loss.
        """
        return self.ce(preds, target)

    def branch_loss(self, branches: list, target: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised loss across all auxiliary branches.

        Each branch is trained independently against ground truth.

        Args:
            branches (list[Tensor]): List of branch logits.
            target (Tensor): Ground-truth labels.

        Returns:
            Tensor: Summed branch loss.
        """

        total_loss = 0.0

        for pred in branches:
            total_loss += self.classification_loss(pred, target)

        return total_loss

    def distillation_loss(self, branches: list) -> torch.Tensor:
        """
        Compute pairwise knowledge distillation between all branches.

        Each branch learns from every other branch using KL divergence.

        Args:
            branches (list[Tensor]): List of branch logits.

        Returns:
            Tensor: Average pairwise KD loss.
        """

        total_loss = 0.0
        count = 0

        # Pairwise KD: (i, j) for all unique combinations
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                total_loss += self.kd(branches[i], branches[j])
                count += 1

        # Avoid division by zero
        return total_loss / max(count, 1)

    def forward(self, outputs: dict, target: torch.Tensor) -> dict:
        """
        Compute full multi-view loss.

        Expected outputs dictionary format:
        {
            "fusion": Tensor,  # main prediction head
            "aps": Tensor,     # branch 1
            "noise": Tensor,   # branch 2
            "glcm": Tensor,    # branch 3
            "dct": Tensor      # branch 4
        }

        Args:
            outputs (dict): Model outputs from multiple branches.
            target (Tensor): Ground-truth labels.

        Returns:
            dict: Dictionary of total and component losses.
        """

        # Main fusion prediction loss (primary objective)
        fusion_pred = outputs["fusion"]
        fusion_loss = self.classification_loss(fusion_pred, target)

        # Collect all branch predictions
        branch_preds = [
            outputs["aps"],
            outputs["noise"],
            outputs["glcm"],
            outputs["dct"],
        ]

        # Supervised loss for each branch
        branch_loss = self.branch_loss(branch_preds, target)

        # Knowledge distillation between branches
        kd_loss = self.distillation_loss(branch_preds)

        # Weighted combination of all losses
        total_loss = fusion_loss + self.alpha * branch_loss + self.beta * kd_loss

        # Return structured logging dictionary
        return {
            "total_loss": total_loss,
            "fusion_loss": fusion_loss,
            "branch_loss": branch_loss,
            "kd_loss": kd_loss,
        }
