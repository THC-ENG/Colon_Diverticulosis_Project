import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SANET_SRC = REPO_ROOT / "external_models" / "SANet" / "src"
if str(SANET_SRC) not in sys.path:
    sys.path.append(str(SANET_SRC))

from res2net import Res2Net  # noqa: E402


def _init_head(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SANet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        norm_type: str = "bn",
        pretrained_path: str = "checkpoints/res2net50_v1b_26w_4s-3cf99910.pth",
    ):
        super().__init__()
        if int(in_channels) != 3 or int(num_classes) != 1:
            raise ValueError("SANet baseline currently supports RGB input and binary output only.")

        c = int(base_channels)
        self.bkbone = Res2Net([3, 4, 6, 3], str(REPO_ROOT / pretrained_path))
        self._load_backbone(pretrained_path)
        self.linear5 = nn.Sequential(nn.Conv2d(2048, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(512, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(c * 3, 1, 1)
        _init_head(nn.Sequential(self.linear5, self.linear4, self.linear3, self.predict))

    def _load_backbone(self, pretrained_path: str):
        path = REPO_ROOT / pretrained_path
        if not path.exists():
            raise FileNotFoundError(f"SANet Res2Net pretrained checkpoint not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        self.bkbone.load_state_dict(state, strict=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        _, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.shape[-2:], mode="bilinear", align_corners=True)
        out4 = F.interpolate(out4, size=out3.shape[-2:], mode="bilinear", align_corners=True)
        pred = torch.cat([out5, out4 * out5, out3 * out4 * out5], dim=1)
        pred = self.predict(pred)
        return F.interpolate(pred, size=input_size, mode="bilinear", align_corners=True)
