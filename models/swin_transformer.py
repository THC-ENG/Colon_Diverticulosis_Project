import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True,
        )
        self.window_size = window_size

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False)[0]


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.window_size = window_size
        self.shift = shift

    def forward(self, x):
        B, H, W, C = x.shape

        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(self.norm1(x_windows))
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift:
            shift_size = self.window_size // 2
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        depth=2,
        window_size=8,
        use_shifted_window=True,
    ):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim=out_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(use_shifted_window and (i % 2 == 1)),
                )
            )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.downsample(x)
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
