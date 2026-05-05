from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.architectures import build_model


def build(in_channels: int = 3, n_classes: int = 1):
    return build_model("rmau_net", in_channels=in_channels, n_classes=n_classes)
