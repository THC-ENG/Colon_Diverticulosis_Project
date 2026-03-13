import torch
import torch.nn as nn

from .attention_gate import AttentionGate
from .resnet_encoder import ResNetShallowEncoder
from .swin_transformer import SwinStage


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(out_channels, skip_channels, inter_channels=max(out_channels // 2, 16))
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResSwinUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_pretrained: bool = True,
        swin_heads_stage3: int = 8,
        swin_heads_stage4: int = 8,
        swin_depth_stage3: int = 2,
        swin_depth_stage4: int = 2,
        use_boundary: bool = False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Current ResNet encoder implementation expects in_channels=3.")

        self.use_boundary = use_boundary
        self.encoder = ResNetShallowEncoder(pretrained=encoder_pretrained)

        self.swin_stage3 = SwinStage(
            in_channels=128,
            out_channels=256,
            num_heads=swin_heads_stage3,
            depth=swin_depth_stage3,
        )
        self.swin_stage4 = SwinStage(
            in_channels=256,
            out_channels=512,
            num_heads=swin_heads_stage4,
            depth=swin_depth_stage4,
        )

        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(32, 1, kernel_size=1) if use_boundary else None

    def forward(self, x: torch.Tensor):
        x0, x1, x2 = self.encoder(x)
        x3 = self.swin_stage3(x2)
        x4 = self.swin_stage4(x3)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        final_feat = self.final_up(d1)
        seg_logits = self.seg_head(final_feat)

        if not self.use_boundary:
            return seg_logits

        boundary_logits = self.boundary_head(final_feat)
        return seg_logits, boundary_logits
