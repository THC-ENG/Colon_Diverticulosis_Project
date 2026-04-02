import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_gate import AttentionGate
from .resnet_encoder import ResNetShallowEncoder
from .swin_transformer import SwinStage


def _make_norm2d(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str = "bn"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm2d(out_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm2d(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_type: str = "bn"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(out_channels, skip_channels, inter_channels=max(out_channels // 2, 16))
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, norm_type=norm_type)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    b, c, h, w = x.shape
    orig_hw = (h, w)
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh, orig_hw


def _haar_idwt2d(
    ll: torch.Tensor,
    lh: torch.Tensor,
    hl: torch.Tensor,
    hh: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    b, c, h, w = ll.shape
    x = torch.zeros((b, c, h * 2, w * 2), dtype=ll.dtype, device=ll.device)

    x[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    x[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    x[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    x[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return x[:, :, : out_hw[0], : out_hw[1]]


class HighFrequencyRefiner(nn.Module):
    def __init__(self, channels: int, norm_type: str = "bn"):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.norm1 = _make_norm2d(channels, norm_type)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm2 = _make_norm2d(channels, norm_type)
        self.expand = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.reduce(x)
        x = self.act(self.norm1(x))
        x = self.pw(self.dw(x))
        x = self.act(self.norm2(x))
        x = self.expand(x)
        return x + residual


class WaveletDecoupledBottleneck(nn.Module):
    def __init__(self, channels: int, norm_type: str = "bn"):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_norm2d(channels, norm_type),
            nn.GELU(),
        )
        self.hf_refiner = HighFrequencyRefiner(channels=channels, norm_type=norm_type)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_norm2d(channels, norm_type),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh, out_hw = _haar_dwt2d(x)
        ll = self.low_proj(ll)
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.hf_refiner(hf)
        lh, hl, hh = torch.chunk(hf, chunks=3, dim=1)
        rec = _haar_idwt2d(ll, lh, hl, hh, out_hw=out_hw)
        return x + self.fuse(rec)


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
        norm_type: str = "bn",
        deep_supervision: bool = False,
        window_size: int = 8,
        use_shift_mask: bool = True,
        use_rel_pos_bias: bool = True,
        pad_to_window: bool = True,
        use_wavelet_bottleneck: bool = False,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Current ResNet encoder implementation expects in_channels=3.")

        self.use_boundary = use_boundary
        self.deep_supervision = deep_supervision
        self.use_wavelet_bottleneck = use_wavelet_bottleneck
        self.encoder = ResNetShallowEncoder(pretrained=encoder_pretrained)

        self.swin_stage3 = SwinStage(
            in_channels=128,
            out_channels=256,
            num_heads=swin_heads_stage3,
            depth=swin_depth_stage3,
            window_size=window_size,
            use_shift_mask=use_shift_mask,
            use_rel_pos_bias=use_rel_pos_bias,
            pad_to_window=pad_to_window,
            norm_type=norm_type,
        )
        self.swin_stage4 = SwinStage(
            in_channels=256,
            out_channels=512,
            num_heads=swin_heads_stage4,
            depth=swin_depth_stage4,
            window_size=window_size,
            use_shift_mask=use_shift_mask,
            use_rel_pos_bias=use_rel_pos_bias,
            pad_to_window=pad_to_window,
            norm_type=norm_type,
        )
        self.wavelet_bottleneck = (
            WaveletDecoupledBottleneck(channels=512, norm_type=norm_type)
            if use_wavelet_bottleneck
            else None
        )

        self.dec4 = UpBlock(512, 256, 256, norm_type=norm_type)
        self.dec3 = UpBlock(256, 128, 128, norm_type=norm_type)
        self.dec2 = UpBlock(128, 64, 64, norm_type=norm_type)
        self.dec1 = UpBlock(64, 64, 64, norm_type=norm_type)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(32, 1, kernel_size=1) if use_boundary else None

        if self.deep_supervision:
            self.aux_head_d2 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(128, num_classes, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0, x1, x2 = self.encoder(x)
        x3 = self.swin_stage3(x2)
        x4 = self.swin_stage4(x3)
        if self.wavelet_bottleneck is not None:
            x4 = self.wavelet_bottleneck(x4)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        final_feat = self.final_up(d1)
        seg_logits = self.seg_head(final_feat)

        if not self.deep_supervision and not self.use_boundary:
            return seg_logits

        boundary_logits = self.boundary_head(final_feat) if self.use_boundary else None

        if not self.deep_supervision:
            return seg_logits, boundary_logits

        out_size = seg_logits.shape[-2:]
        aux_logits = [
            F.interpolate(self.aux_head_d2(d2), size=out_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head_d3(d3), size=out_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head_d4(d4), size=out_size, mode="bilinear", align_corners=False),
        ]
        return {
            "seg": seg_logits,
            "boundary": boundary_logits,
            "aux": aux_logits,
        }
