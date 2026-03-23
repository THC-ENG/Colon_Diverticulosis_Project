import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_gate import AttentionGate
from .resnet_encoder import ResNetShallowEncoder
from .swin_transformer import SwinStage


def _make_norm2d(channels: int, norm_type: str) -> nn.Module:  #归一化函数，支持 BN/GN 切换
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):  #标准双卷积块
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
        skip = self.attn(x, skip)  #注意力门控，用解码器当前语义信息和编码器细节特征共同决定该关注哪里
        x = torch.cat([x, skip], dim=1)  #通道维拼接，拼接后送入卷积块融合特征
        return self.conv(x)  #融合后特征继续送入下一个上采样块


class ResSwinUNet(nn.Module):  # ResSwinUNet整体架构，ResNet34 作为编码器提取多尺度特征，Swin Transformer 作为桥接模块增强全局上下文信息，UNet 解码器逐步融合特征并恢复空间分辨率
    def __init__(
        self,
        in_channels: int = 3,  #输入图像通道数，当前 ResNet 编码器实现要求必须为 3
        num_classes: int = 1,  #分割类别数
        encoder_pretrained: bool = True,  #是否使用预训练的 ResNet34 权重
        swin_heads_stage3: int = 8,  #Swin Transformer 第3阶段的注意力头数
        swin_heads_stage4: int = 8,  #Swin Transformer 第4阶段的注意力头数
        swin_depth_stage3: int = 2,  #Swin Transformer 第3阶段的SwinBlock块数
        swin_depth_stage4: int = 2,  #Swin Transformer 第4阶段的SwinBlock块数
        use_boundary: bool = False,  #是否使用边界检测分支
        norm_type: str = "bn",  #归一化类型，支持 "bn"（BatchNorm）和 "gn"（GroupNorm）
        deep_supervision: bool = False,  #是否使用深监督，在中间层添加辅助分割头以增强训练信号
        window_size: int = 8,  #Swin Transformer 的窗口大小
        use_shift_mask: bool = True,  #Swin Transformer 是否使用交错窗口（shifted window）机制以增强跨窗口信息交流
        use_rel_pos_bias: bool = True,  #Swin Transformer 是否使用相对位置偏置以增强位置编码能力
        pad_to_window: bool = True,  #是否在输入特征图尺寸不是窗口大小整数倍时进行填充以适应窗口划分
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Current ResNet encoder implementation expects in_channels=3.")

        self.use_boundary = use_boundary
        self.deep_supervision = deep_supervision
        self.encoder = ResNetShallowEncoder(pretrained=encoder_pretrained)  #ResNet34 作为编码器提取多尺度特征，输出三个阶段的特征图，分别是 1/2、1/4、1/8 分辨率

        self.swin_stage3 = SwinStage(  #x2 -> swin_stage3 -> x3
            in_channels=128,  #ResNet 输出的第3阶段特征图通道数是128, 和 ResNetShallowEncoder 输出的 x2 通道数一致
            out_channels=256,  #Swin Transformer 第3阶段输出通道数256, 也是第4阶段输入通道数
            num_heads=swin_heads_stage3,
            depth=swin_depth_stage3,
            window_size=window_size,
            use_shift_mask=use_shift_mask,
            use_rel_pos_bias=use_rel_pos_bias,
            pad_to_window=pad_to_window,
            norm_type=norm_type,
        )
        self.swin_stage4 = SwinStage(  #x3 -> swin_stage4 -> x4
            in_channels=256,  #Swin Transformer 第3阶段输出通道数256, 也是第4阶段输入通道数
            out_channels=512,  #Swin Transformer 第4阶段输出通道数512, 也是解码器第1个上采样块输入通道数
            num_heads=swin_heads_stage4,
            depth=swin_depth_stage4,
            window_size=window_size,
            use_shift_mask=use_shift_mask,
            use_rel_pos_bias=use_rel_pos_bias,
            pad_to_window=pad_to_window,
            norm_type=norm_type,
        )

        # x4 上采样 + x3 skip → d4
        self.dec4 = UpBlock(512, 256, 256, norm_type=norm_type)  #解码器第1个上采样块，输入通道512（来自 swin_stage4 输出），skip 通道256（来自 swin_stage3 输出），输出通道256
        # d4 上采样 + x2 skip → d3
        self.dec3 = UpBlock(256, 128, 128, norm_type=norm_type)  #解码器第2个上采样块，输入通道256（来自 dec4 输出），skip 通道128（来自 ResNetShallowEncoder 输出的 x2），输出通道128
        # d3 上采样 + x1 skip → d2
        self.dec2 = UpBlock(128, 64, 64, norm_type=norm_type)  #解码器第3个上采样块，输入通道128（来自 dec3 输出），skip 通道64（来自 ResNetShallowEncoder 输出的 x1），输出通道64
        # d2 上采样 + x0 skip → d1
        self.dec1 = UpBlock(64, 64, 64, norm_type=norm_type)  #解码器第4个上采样块，输入通道64（来自 dec2 输出），skip 通道64（来自 ResNetShallowEncoder 输出的 x0），输出通道64
        # d1 -> final_up -> final_feat
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  #最后上采样一层，把通道数从64降到32，分辨率恢复到输入图像大小
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)  #分割头，1x1卷积把通道数从32变成 num_classes，输出分割概率图
        self.boundary_head = nn.Conv2d(32, 1, kernel_size=1) if use_boundary else None  #边界检测头，1x1卷积把通道数从32变成1，输出边界概率图

        if self.deep_supervision:  #如果使用深监督，在解码器中间层添加辅助分割头
            self.aux_head_d2 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.aux_head_d3 = nn.Conv2d(128, num_classes, kernel_size=1)
            self.aux_head_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0, x1, x2 = self.encoder(x) #1/2, 64 -> 1/4, 64 -> 1/8, 128
        x3 = self.swin_stage3(x2) #1/8, 256 -> swin_stage3 -> 1/16, 256
        x4 = self.swin_stage4(x3) #1/16, 256 -> swin_stage4 -> 1/32, 512

        d4 = self.dec4(x4, x3) #1/32, 512 + 1/16, 256 -> dec4 -> 1/16, 256
        d3 = self.dec3(d4, x2) #1/16, 256 + 1/8, 128 -> dec3 -> 1/8, 128
        d2 = self.dec2(d3, x1) #1/8, 128 + 1/4, 64 -> dec2 -> 1/4, 64
        d1 = self.dec1(d2, x0) #1/4, 64 + 1/2, 64 -> dec1 -> 1/2, 64

        final_feat = self.final_up(d1) #1/2, 64 -> final_up -> 1, 32
        seg_logits = self.seg_head(final_feat)

        if not self.deep_supervision and not self.use_boundary:
            return seg_logits

        boundary_logits = self.boundary_head(final_feat) if self.use_boundary else None

        if not self.deep_supervision:
            return seg_logits, boundary_logits

        out_size = seg_logits.shape[-2:] #深监督输出需要上采样到和最终分割图一样的尺寸，计算出最终分割图的空间尺寸
        aux_logits = [
            F.interpolate(self.aux_head_d2(d2), size=out_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head_d3(d3), size=out_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux_head_d4(d4), size=out_size, mode="bilinear", align_corners=False),
        ]
        return {   #返回一个字典，包含最终分割图、边界图（如果有）和辅助分割图（如果有）
            "seg": seg_logits,
            "boundary": boundary_logits,
            "aux": aux_logits,
        }
