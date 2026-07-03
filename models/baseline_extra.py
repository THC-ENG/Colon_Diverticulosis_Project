import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

try:
    import timm
except ModuleNotFoundError:  # pragma: no cover
    timm = None


def _norm(num_channels: int, norm_type: str = "gn") -> nn.Module:
    norm = str(norm_type).lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    groups = min(8, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1, stride: int = 1, norm_type: str = "gn"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False),
            _norm(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm_type: str = "gn"):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=1, norm_type=norm_type)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _norm(out_channels, norm_type),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(out_channels, norm_type),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates: tuple[int, ...] = (1, 6, 12, 18), norm_type: str = "gn"):
        super().__init__()
        branches = []
        for rate in rates:
            if rate == 1:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0, norm_type=norm_type))
            else:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, norm_type=norm_type))
        self.branches = nn.ModuleList(branches)
        self.project = ConvBNReLU(out_channels * len(branches), out_channels, kernel_size=1, padding=0, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c1, c2, c3, c4, c5


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm_type: str = "gn"):
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("DeepLabV3Plus baseline expects RGB input.")
        decoder_channels = max(128, int(base_channels) * 4)
        low_channels = max(32, int(base_channels))
        self.encoder = ResNet50Encoder()
        self.aspp = ASPP(2048, decoder_channels, norm_type=norm_type)
        self.low_proj = ConvBNReLU(256, low_channels, kernel_size=1, padding=0, norm_type=norm_type)
        self.decoder = nn.Sequential(
            ConvBNReLU(decoder_channels + low_channels, decoder_channels, norm_type=norm_type),
            ConvBNReLU(decoder_channels, decoder_channels, norm_type=norm_type),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _c1, c2, _c3, _c4, c5 = self.encoder(x)
        high = self.aspp(c5)
        high = F.interpolate(high, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(c2)
        out = self.decoder(torch.cat([high, low], dim=1))
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class FPNResNet50(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm_type: str = "gn"):
        super().__init__()
        if int(in_channels) != 3:
            raise ValueError("FPNResNet50 baseline expects RGB input.")
        fpn_channels = max(128, int(base_channels) * 4)
        self.encoder = ResNet50Encoder()
        self.lateral2 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, fpn_channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(2048, fpn_channels, kernel_size=1)
        self.smooth2 = ConvBNReLU(fpn_channels, fpn_channels, norm_type=norm_type)
        self.smooth3 = ConvBNReLU(fpn_channels, fpn_channels, norm_type=norm_type)
        self.smooth4 = ConvBNReLU(fpn_channels, fpn_channels, norm_type=norm_type)
        self.head = nn.Sequential(
            ConvBNReLU(fpn_channels * 4, fpn_channels, norm_type=norm_type),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _c1, c2, c3, c4, c5 = self.encoder(x)
        p5 = self.lateral5(c5)
        p4 = self.smooth4(self.lateral4(c4) + self._up_to(p5, c4))
        p3 = self.smooth3(self.lateral3(c3) + self._up_to(p4, c3))
        p2 = self.smooth2(self.lateral2(c2) + self._up_to(p3, c2))
        feats = [p2, self._up_to(p3, p2), self._up_to(p4, p2), self._up_to(p5, p2)]
        out = self.head(torch.cat(feats, dim=1))
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class ResUNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm_type: str = "gn"):
        super().__init__()
        c = int(base_channels)
        self.enc1 = ResidualBlock(in_channels, c, norm_type=norm_type)
        self.enc2 = ResidualBlock(c, c * 2, stride=2, norm_type=norm_type)
        self.enc3 = ResidualBlock(c * 2, c * 4, stride=2, norm_type=norm_type)
        self.enc4 = ResidualBlock(c * 4, c * 8, stride=2, norm_type=norm_type)
        self.bridge = ASPP(c * 8, c * 16, rates=(1, 3, 6, 9), norm_type=norm_type)
        self.dec4 = ResidualBlock(c * 16 + c * 8, c * 8, norm_type=norm_type)
        self.dec3 = ResidualBlock(c * 8 + c * 4, c * 4, norm_type=norm_type)
        self.dec2 = ResidualBlock(c * 4 + c * 2, c * 2, norm_type=norm_type)
        self.dec1 = ResidualBlock(c * 2 + c, c, norm_type=norm_type)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bridge(e4)
        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        return self.head(d1)


class SegFormerB0(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32, norm_type: str = "gn"):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required for the SegFormer-B0-style baseline.")
        if int(in_channels) != 3:
            raise ValueError("SegFormerB0 baseline expects RGB input.")
        decoder_channels = max(128, int(base_channels) * 4)
        self.encoder = timm.create_model("pvt_v2_b0", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        channels = self.encoder.feature_info.channels()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, decoder_channels, kernel_size=1, bias=False), _norm(decoder_channels, norm_type), nn.ReLU(inplace=True))
            for ch in channels
        ])
        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_channels * len(channels), decoder_channels, kernel_size=1, padding=0, norm_type=norm_type),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        target_size = feats[0].shape[-2:]
        projected = [
            F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.proj, feats)
        ]
        out = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
