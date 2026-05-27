"""
Multi-view loss combining supervised classification on a fusion head and multiple branches,
plus cross-branch knowledge distillation.

It sums:
1) Fusion cross-entropy loss (main prediction),
2) Branch cross-entropy losses averaged (auxiliary supervision),
3) Pairwise KL-based distillation between branches to enforce consistency (temperature scaled)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Apply temperature scaling:
        # Student uses log_softmax
        # Teacher uses softmax to form target probability distribution
        student_log_probs = F.log_softmax(student_logits / t, dim=1)
        teacher_probs = F.softmax(teacher_logits / t, dim=1)

        # KL divergence between teacher (target) and student (input)
        # Scaled by t^2 to maintain consistent gradient magnitude across temperatures
        return self.kl(student_log_probs, teacher_probs) * (t**2)


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

    def forward(self, outputs: dict, target: torch.Tensor) -> dict:
        """
        Compute full multi-view loss by safely unpacking nested dictionaries.
        """
        # 1. Primary fusion prediction loss (main head objective)
        fusion_pred = outputs["fusion"]
        fusion_loss = self.classification_loss(fusion_pred, target)

        total_branch_loss = 0.0
        kd_loss = 0.0

        # 2. Extract logits safely from the nested "branches" sub-dictionary
        if "branches" in outputs and isinstance(outputs["branches"], dict):
            branch_dict = outputs["branches"]

            # Convert the active logits dictionary values directly into a clean list of tensors
            branch_preds = list(branch_dict.values())
            num_branches = len(branch_preds)

            if num_branches > 0:
                # Calculate supervised cross-entropy for each active branch
                for pred in branch_preds:
                    total_branch_loss += self.classification_loss(pred, target)

                # Average the auxiliary loss across the number of branches for training stability
                total_branch_loss = total_branch_loss / num_branches

                # 3. Calculate cross-branch mutual learning knowledge distillation
                total_kd_loss = 0.0
                kd_count = 0

                # Pairwise bidirectional loops over the active branch tensors
                for i in range(num_branches):
                    for j in range(num_branches):
                        if i != j:  # Cross-distill symmetrically: i -> j AND j -> i
                            total_kd_loss += self.kd(branch_preds[i], branch_preds[j])
                            kd_count += 1

                if kd_count > 0:
                    kd_loss = total_kd_loss / kd_count

        # 4. Final Weighted Combination
        total_loss = (
            fusion_loss + (self.alpha * total_branch_loss) + (self.beta * kd_loss)
        )

        # Return structured metrics matching your exact dictionary tracking keys
        return {
            "total_loss": total_loss,
            "fusion_loss": fusion_loss,
            "branch_loss": total_branch_loss,
            "kd_loss": kd_loss,
        }
