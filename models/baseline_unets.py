import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline_pranet import PraNet
from .baseline_sanet import SANet
from .baseline_transunet import TransUNet
from .baseline_extra import DeepLabV3Plus, FPNResNet50, ResUNetPlusPlus, SegFormerB0


def _norm(num_channels: int, norm_type: str = "gn") -> nn.Module:
    norm = str(norm_type).lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    groups = min(8, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "gn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _norm(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        norm_type: str = "gn",
    ):
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock(in_channels, c, norm_type)
        self.enc2 = ConvBlock(c, c * 2, norm_type)
        self.enc3 = ConvBlock(c * 2, c * 4, norm_type)
        self.enc4 = ConvBlock(c * 4, c * 8, norm_type)
        self.bottleneck = ConvBlock(c * 8, c * 16, norm_type)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8, norm_type)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4, norm_type)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, norm_type)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, norm_type)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        norm_type: str = "gn",
    ):
        super().__init__()
        c = int(base_channels)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ConvBlock(in_channels, c, norm_type)
        self.conv1_0 = ConvBlock(c, c * 2, norm_type)
        self.conv2_0 = ConvBlock(c * 2, c * 4, norm_type)
        self.conv3_0 = ConvBlock(c * 4, c * 8, norm_type)
        self.conv4_0 = ConvBlock(c * 8, c * 16, norm_type)

        self.conv0_1 = ConvBlock(c + c * 2, c, norm_type)
        self.conv1_1 = ConvBlock(c * 2 + c * 4, c * 2, norm_type)
        self.conv2_1 = ConvBlock(c * 4 + c * 8, c * 4, norm_type)
        self.conv3_1 = ConvBlock(c * 8 + c * 16, c * 8, norm_type)

        self.conv0_2 = ConvBlock(c * 2 + c * 2, c, norm_type)
        self.conv1_2 = ConvBlock(c * 2 * 2 + c * 4, c * 2, norm_type)
        self.conv2_2 = ConvBlock(c * 4 * 2 + c * 8, c * 4, norm_type)

        self.conv0_3 = ConvBlock(c * 3 + c * 2, c, norm_type)
        self.conv1_3 = ConvBlock(c * 2 * 3 + c * 4, c * 2, norm_type)

        self.conv0_4 = ConvBlock(c * 4 + c * 2, c, norm_type)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))
        return self.head(x0_4)


def build_baseline_model(model_name: str, **kwargs) -> nn.Module:
    name = str(model_name).strip().lower().replace("-", "").replace("_", "")
    if name in {"unet", "u"}:
        return UNet(**kwargs)
    if name in {"unet++", "unetplusplus", "nestedunet", "unetpp"}:
        return UNetPlusPlus(**kwargs)
    if name in {"pranet", "pra"}:
        return PraNet(**kwargs)
    if name in {"transunet", "transunetvit", "transunetr50vitb16"}:
        return TransUNet(**kwargs)
    if name in {"sanet", "shallowattentionnetwork"}:
        return SANet(**kwargs)
    if name in {"deeplabv3plus", "deeplabv3+", "deeplab"}:
        return DeepLabV3Plus(**kwargs)
    if name in {"fpnresnet50", "fpn", "fpnr50"}:
        return FPNResNet50(**kwargs)
    if name in {"resunet++", "resunetplusplus", "resunetpp"}:
        return ResUNetPlusPlus(**kwargs)
    if name in {"segformerb0", "segformer_b0", "segformer"}:
        return SegFormerB0(**kwargs)
    raise ValueError(f"Unsupported local baseline model: {model_name}")
