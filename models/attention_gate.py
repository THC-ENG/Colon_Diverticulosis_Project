import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Attention gate 用于 skip connections."""

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.w_g = nn.Sequential(  #把解码器特征映射到中间维度
            nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_x = nn.Sequential(  #skip 特征也映射到同样中间维度
            nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(  #生成注意力图
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.relu(self.w_g(g) + self.w_x(x))) #用解码器当前语义信息和编码器细节特征共同决定该关注哪里
        return x * alpha  #[B, C, H, W] * [B, 1, H, W]
