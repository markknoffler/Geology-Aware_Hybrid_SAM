import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        target = target.float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)

        tp = (probs * target).sum()
        fp = ((1.0 - target) * probs).sum()
        fn = (target * (1.0 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class SegmentationLoss(nn.Module):
    """
    Shared loss wrapper for all baselines.
    Supports:
      - single output: logits
      - deep supervision output: (main, aux2, aux3, ...)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        main_weight: float = 1.0,
        aux2_weight: float = 0.6,
        aux3_weight: float = 0.4,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux2_weight = aux2_weight
        self.aux3_weight = aux3_weight
        self.criterion = TverskyLoss(alpha=alpha, beta=beta)

    @staticmethod
    def _resize_target(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.shape[-2:] != pred.shape[-2:]:
            target = F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest")
        return target

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, (tuple, list)):
            main = outputs[0]
            aux2 = outputs[1] if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor) else None
            aux3 = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], torch.Tensor) else None
        else:
            main, aux2, aux3 = outputs, None, None

        t_main = self._resize_target(target, main)
        total = self.main_weight * self.criterion(main, t_main)

        if aux2 is not None:
            t_aux2 = self._resize_target(target, aux2)
            total = total + self.aux2_weight * self.criterion(aux2, t_aux2)
        if aux3 is not None:
            t_aux3 = self._resize_target(target, aux3)
            total = total + self.aux3_weight * self.criterion(aux3, t_aux3)
        return total
