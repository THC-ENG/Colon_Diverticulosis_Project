import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm2d(channels: int, norm_type: str) -> nn.Module:  #支持 BN/GN 切换
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        groups = min(32, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def window_partition(x, window_size):  #把特征图切成窗口
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):  #把窗口拼回原图
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):  #只在每个窗口内部做注意力，不做全局注意力
    def __init__(self, dim, num_heads, window_size, use_rel_pos_bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.use_rel_pos_bias = use_rel_pos_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        if self.use_rel_pos_bias:
            bias_size = (2 * window_size - 1) * (2 * window_size - 1)
            self.relative_position_bias_table = nn.Parameter(torch.zeros(bias_size, num_heads))
            self.register_buffer(
                "relative_position_index",
                self._build_relative_position_index(window_size),
                persistent=False,
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.relative_position_bias_table = None
            self.register_buffer("relative_position_index", torch.empty(0), persistent=False)

    @staticmethod
    def _build_relative_position_index(window_size: int) -> torch.Tensor:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        return relative_coords.sum(-1) #[N, N]，这里 N = window_size * window_size

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)    #最前面那一维长度是 3，分别对应 q、k、v
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)   #计算窗口里每个token对其他token的注意力分数

        if self.use_rel_pos_bias: #加相对位置偏置
            rel_pos = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rel_pos = rel_pos.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()  #[num_heads, N, N]
            attn = attn + rel_pos.unsqueeze(0)   #给原始注意力分数加上相对位置偏置

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)  #得到注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module): #本质是：LN → Attention → Residual → LN → MLP → Residual
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift=False,
        use_shift_mask=True,
        use_rel_pos_bias=True,
        pad_to_window=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            num_heads,
            window_size,
            use_rel_pos_bias=use_rel_pos_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(      #前馈网络
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.use_shift_mask = use_shift_mask
        self.pad_to_window = pad_to_window
        self._attn_mask_cache = {}

    #用 mask 屏蔽掉不同窗口之间的注意力连接，确保每个 token 只能 attend 同一窗口内的 token
    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (  #把图像在高和宽上都分成 3 段
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # 计算窗口内任意两个 token 的 mask
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #变成真正的注意力 mask，同一窗口 mask 值为 0，不同窗口为-100
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        _, H, W, _ = x.shape   #[B, H, W, C]
        shortcut = x
        x = self.norm1(x)   #先做第一层 LayerNorm

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if (pad_h > 0 or pad_w > 0) and not self.pad_to_window:
            raise ValueError(
                f"Feature map {(H, W)} is not divisible by window_size={self.window_size}. "
                "Set pad_to_window=True to enable auto padding."
            )
        #padding
        if pad_h > 0 or pad_w > 0:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1).contiguous()

        Hp, Wp = x.shape[1], x.shape[2]
        attn_mask = None

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.use_shift_mask:
                attn_mask = self._get_attn_mask(Hp, Wp, x.device)

        x_windows = window_partition(x, self.window_size)
        if attn_mask is not None and attn_mask.dtype != x_windows.dtype:
            attn_mask = attn_mask.to(dtype=x_windows.dtype)
        attn_windows = self.attn(x_windows, mask=attn_mask)   #窗口注意力
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  #[B, Hp, Wp, C]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_h > 0 or pad_w > 0:  #去padding
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  #LayerNorm -> MLP -> residual
        return x   #[B, H, W, C]


class SwinStage(nn.Module):   #先做一次下采样，再堆若干个 SwinBlock
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        depth=2,
        window_size=8,
        use_shifted_window=True,
        use_shift_mask=True,
        use_rel_pos_bias=True,
        pad_to_window=True,
        norm_type="bn",
    ):
        super().__init__()
        #下采样层，把输入特征图的分辨率缩小一半，通道数翻倍
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            _make_norm2d(out_channels, norm_type),
            nn.ReLU(inplace=True),
        )
        #构造多个 block
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinBlock(
                    dim=out_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(use_shifted_window and (i % 2 == 1)), #交替使用普通窗口和移位窗口
                    use_shift_mask=use_shift_mask,
                    use_rel_pos_bias=use_rel_pos_bias,
                    pad_to_window=pad_to_window,
                )
            )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.downsample(x)  #[B, C, H, W]
        x = x.permute(0, 2, 3, 1) #[B, C, H, W] -> [B, H, W, C]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  #[B, C, H, W]
        return x
