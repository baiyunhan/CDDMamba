"""Joint segmentation and CDFM supervision for the proposed network."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ProposedJointLoss(nn.Module):
    """L_total = L_CE + lambda_Dice * L_Dice + lambda_FM * L_FM."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        flow_weight: float = 0.1,
        aux_weight: float = 0.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.flow_weight = flow_weight
        self.aux_weight = aux_weight
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _dice_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        probabilities = logits.softmax(dim=1)
        valid = target.ne(self.ignore_index)
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)
        valid_mask = valid.unsqueeze(1).to(probabilities.dtype)
        probabilities = probabilities * valid_mask
        one_hot = one_hot * valid_mask

        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # Do not reward a class that is absent from the current batch.
        present = one_hot.sum(dim=dimensions) > 0
        if present.any():
            return 1.0 - dice[present].mean()
        return logits.new_zeros(())

    def forward(self, output: Tensor | Mapping[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(output, Mapping):
            logits = output["logits"]
            flow_loss = output.get("flow_loss", logits.new_zeros(()))
            aux_logits = output.get("aux_logits")
        else:
            logits = output
            flow_loss = logits.new_zeros(())
            aux_logits = None

        # CrossEntropyLoss(reduction="mean") is NaN when every pixel is ignored.
        # Such a crop can occur after random scaling/cropping, so keep a
        # differentiable zero instead of poisoning the whole optimisation step.
        if target.ne(self.ignore_index).any():
            ce_loss = self.cross_entropy(logits, target)
        else:
            ce_loss = logits.sum() * 0.0
        dice_loss = self._dice_loss(logits, target)
        aux_loss = logits.new_zeros(())
        if aux_logits is not None and self.aux_weight > 0:
            if target.ne(self.ignore_index).any():
                aux_ce = self.cross_entropy(aux_logits, target)
            else:
                aux_ce = aux_logits.sum() * 0.0
            aux_dice = self._dice_loss(aux_logits, target)
            aux_loss = aux_ce + self.dice_weight * aux_dice

        total_loss = (
            ce_loss
            + self.dice_weight * dice_loss
            + self.flow_weight * flow_loss
            + self.aux_weight * aux_loss
        )
        self.last_components = {
            "ce_loss": ce_loss.detach(),
            "dice_loss": dice_loss.detach(),
            "flow_loss": flow_loss.detach(),
            "aux_loss": aux_loss.detach(),
        }
        return total_loss
