from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.architectures import build_model


def build(n_classes: int = 1):
    return build_model("dual_stream_unet", in_channels=3, n_classes=n_classes)
