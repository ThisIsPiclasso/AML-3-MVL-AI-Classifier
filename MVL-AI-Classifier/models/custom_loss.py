"""
Dynamic multi-view loss.

Combines:
1) Fusion classification loss
2) Auxiliary branch classification losses
3) Cross-branch knowledge distillation

Expected model output format:
{
    "fusion": Tensor,
    "branches": [Tensor, Tensor, ...],
    "embeddings": [...]
}

The loss automatically supports any number of branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from MVL_AI_Classifier.constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_TEMPERATURE,
)


class DistillationLoss(nn.Module):
    """
    KL-divergence based knowledge distillation loss.

    Encourages one branch to match the softened
    probability distribution of another branch.
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()

        self.temperature = temperature

        # KL divergence expects:
        # input  -> log probabilities
        # target -> probabilities
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temperature-scaled KD loss.

        Args:
            student_logits (Tensor):
                Student branch logits.

            teacher_logits (Tensor):
                Teacher branch logits.

        Returns:
            Tensor:
                Scalar distillation loss.
        """

        t = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / t,
            dim=1
        )

        teacher_probs = F.softmax(
            teacher_logits / t,
            dim=1
        )

        # Standard KD scaling correction
        return self.kl(student_log_probs, teacher_probs) * (t ** 2)


class MultiViewLoss(nn.Module):
    """
    Multi-view learning loss supporting dynamic numbers of branches.

    Components:
    - Fusion classification loss
    - Branch classification loss
    - Pairwise branch distillation loss
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Args:
            alpha (float):
                Weight for branch classification loss.

            beta (float):
                Weight for branch distillation loss.

            temperature (float):
                Temperature used in KD.
        """

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = nn.CrossEntropyLoss()
        self.kd = DistillationLoss(temperature)

    def classification_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard cross-entropy classification loss.
        """

        return self.ce(preds, target)

    def branch_loss(
        self,
        branches: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute average supervised loss across all branches.

        Args:
            branches (list[Tensor]):
                Branch logits.

            target (Tensor):
                Ground-truth labels.

        Returns:
            Tensor:
                Average branch classification loss.
        """

        if len(branches) == 0:
            return torch.tensor(0.0, device=target.device)

        total_loss = 0.0

        for branch_logits in branches:
            total_loss += self.classification_loss(
                branch_logits,
                target
            )

        return total_loss / len(branches)

    def distillation_loss(
        self,
        branches: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute pairwise branch knowledge distillation.

        Every branch distills knowledge from every other branch.

        Args:
            branches (list[Tensor]):
                Branch logits.

        Returns:
            Tensor:
                Average pairwise KD loss.
        """

        num_branches = len(branches)

        # Distillation requires at least two branches
        if num_branches < 2:
            return torch.tensor(
                0.0,
                device=branches[0].device if num_branches == 1 else "cpu"
            )

        total_loss = 0.0
        pair_count = 0

        for i in range(num_branches):
            for j in range(i + 1, num_branches):

                # Bidirectional distillation
                total_loss += self.kd(
                    branches[i],
                    branches[j]
                )

                total_loss += self.kd(
                    branches[j],
                    branches[i]
                )

                pair_count += 2

        return total_loss / pair_count

    def forward(
        self,
        outputs: dict,
        target: torch.Tensor,
    ) -> dict:
        """
        Compute full multi-view loss.

        Expected outputs format:
        {
            "fusion": Tensor,
            "branches": [Tensor, Tensor, ...],
            "embeddings": [...]
        }

        Args:
            outputs (dict):
                Model outputs.

            target (Tensor):
                Ground-truth labels.

        Returns:
            dict:
                Structured loss dictionary.
        """

        fusion_logits = outputs["fusion"]
        branch_logits = outputs["branches"]

        # Main fusion supervision
        fusion_loss = self.classification_loss(
            fusion_logits,
            target
        )

        # Auxiliary branch supervision
        branch_loss = self.branch_loss(
            branch_logits,
            target
        )

        # Cross-branch consistency
        kd_loss = self.distillation_loss(
            branch_logits
        )

        # Final weighted loss
        total_loss = (
            fusion_loss
            + self.alpha * branch_loss
            + self.beta * kd_loss
        )

        return {
            "total_loss": total_loss,
            "fusion_loss": fusion_loss,
            "branch_loss": branch_loss,
            "kd_loss": kd_loss,
        }
