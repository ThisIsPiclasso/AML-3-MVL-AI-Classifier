import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """KL distillation"""

    def __init__(self, temperature=2.0):
        super().__init__()

        self.temperature = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):

        t = self.temperature

        student = F.log_softmax(student_logits / t, dim=1)
        teacher = F.softmax(teacher_logits / t, dim=1)

        return self.kl(student, teacher)


class MultiViewLoss(nn.Module):
    """Custom multi-view loss."""

    def __init__(
        self,
        alpha=0.3,
        beta=0.1,
        temperature=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.ce = nn.CrossEntropyLoss()
        self.kd = DistillationLoss(temperature)

    def classification_loss(self, preds, target):
        """CE loss."""

        return self.ce(preds, target)

    def branch_loss(self, branches, target):
        """Branch CE losses."""

        total = 0

        for pred in branches:
            total += self.classification_loss(pred, target)

        return total

    def distillation_loss(self, branches):
        """Branch KD losses."""

        total = 0
        count = 0

        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):

                total += self.kd(branches[i], branches[j])
                count += 1

        return total / count

    def forward(self, outputs, target):
        """
        outputs:
        {
            "fusion": tensor,
            "aps": tensor,
            "noise": tensor,
            "glcm": tensor,
            "dct": tensor
        }
        """

        fusion_pred = outputs["fusion"]

        branch_preds = [
            outputs["aps"],
            outputs["noise"],
            outputs["glcm"],
            outputs["dct"]
        ]

        # main loss
        fusion_loss = self.classification_loss(
            fusion_pred,
            target
        )

        # branch loss
        branch_loss = self.branch_loss(
            branch_preds,
            target
        )

        # kd loss
        kd_loss = self.distillation_loss(
            branch_preds
        )

        # total
        total_loss = (
            fusion_loss +
            self.alpha * branch_loss +
            self.beta * kd_loss
        )

        # logs
        loss_dict = {
            "total_loss": total_loss,
            "fusion_loss": fusion_loss,
            "branch_loss": branch_loss,
            "kd_loss": kd_loss
        }

        return loss_dict