from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.losses import SegmentationLoss

__all__ = ["SegmentationLoss"]
