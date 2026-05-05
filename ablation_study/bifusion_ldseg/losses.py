from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * target, dim=dims)
    den = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


class BiFusionLoss(nn.Module):
    """
    L_total = L_seg(main) + lambda_diff * L_diff + L_deep(aux)
    L_seg = BCE + gamma * Dice
    """

    def __init__(
        self,
        gamma: float = 2.0,
        lambda_diff: float = 1.0,
        deep_weights=(0.5, 0.7, 0.9, 1.0),
    ):
        super().__init__()
        self.gamma = gamma
        self.lambda_diff = lambda_diff
        self.deep_weights = deep_weights
        self.bce = nn.BCEWithLogitsLoss()

    def _seg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
        return self.bce(pred, target) + self.gamma * dice_loss_from_logits(pred, target)

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_main = self._seg_loss(outputs["main"], target)

        deep_preds = [outputs["aux1"], outputs["aux2"], outputs["aux3"], outputs["aux4"]]
        loss_deep = 0.0
        for w, p in zip(self.deep_weights, deep_preds):
            loss_deep = loss_deep + float(w) * self._seg_loss(p, target)

        if "eps_pred" in outputs and "eps_true" in outputs:
            loss_diff = F.mse_loss(outputs["eps_pred"], outputs["eps_true"])
        else:
            loss_diff = torch.tensor(0.0, device=target.device)

        total = loss_main + self.lambda_diff * loss_diff + loss_deep
        return {
            "loss": total,
            "loss_main": loss_main.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_deep": (loss_deep.detach() if isinstance(loss_deep, torch.Tensor) else torch.tensor(loss_deep)),
        }
