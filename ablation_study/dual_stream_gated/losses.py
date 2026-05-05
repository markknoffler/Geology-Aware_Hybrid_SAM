import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1.0 - 1e-4)
        target = target.float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)

        tp = (probs * target).sum()
        fp = ((1.0 - target) * probs).sum()
        fn = (target * (1.0 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class DualStreamLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        main_weight: float = 1.0,
        aux2_weight: float = 0.6,
        aux3_weight: float = 0.4,
        reg_weight: float = 1e-3,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux2_weight = aux2_weight
        self.aux3_weight = aux3_weight
        self.reg_weight = reg_weight
        self.criterion = TverskyLoss(alpha=alpha, beta=beta)

    def _resize_target(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest")
        return target

    def forward(
        self,
        main: torch.Tensor,
        aux2: torch.Tensor,
        aux3: torch.Tensor,
        reg_tuple,
        target: torch.Tensor,
    ):
        target_main = self._resize_target(target, main)
        target_aux2 = self._resize_target(target, aux2)
        target_aux3 = self._resize_target(target, aux3)

        loss_main = self.criterion(main, target_main)
        loss_aux2 = self.criterion(aux2, target_aux2)
        loss_aux3 = self.criterion(aux3, target_aux3)
        reg = self.reg_weight * sum(reg_tuple) if reg_tuple is not None and len(reg_tuple) > 0 else 0.0
        total = (
            self.main_weight * loss_main
            + self.aux2_weight * loss_aux2
            + self.aux3_weight * loss_aux3
            + reg
        )
        reg_value = reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=target.device)
        return {
            "loss": total,
            "loss_main": loss_main.detach(),
            "loss_aux2": loss_aux2.detach(),
            "loss_aux3": loss_aux3.detach(),
            "loss_reg": reg_value.detach(),
        }
