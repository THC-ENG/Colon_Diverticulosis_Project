# SwinStage 与频域解耦细化结构图

> 对应代码：`models/swin_transformer.py`、`models/res_swin_unet.py`

## 参数与尺寸总览（当前 ResSwinUNet 默认）

- `SwinStage3`: `in=128, out=256, depth=2, heads=8, window_size=8, norm_type=bn`
- `SwinStage4`: `in=256, out=512, depth=2, heads=8, window_size=8, norm_type=bn`
- `WaveletDecoupledBottleneck`: `channels=512, norm_type=bn`（仅当 `use_wavelet_bottleneck=True`）
- `_make_norm2d`:
  - `bn` -> `BatchNorm2d(C)`
  - `gn` -> `GroupNorm(groups<=32 且可整除 C, C)`

## 1) SwinStage 细化图（尺寸/层次/归一化）

```mermaid
flowchart TD
    IN3["Stage3 输入 x2\nB x 128 x H/8 x W/8"]
    IN4["Stage4 输入 x3\nB x 256 x H/16 x W/16"]

    DS["Downsample\nConv2d(3x3,s=2,p=1)\n+ Norm2d(bn/gn)\n+ ReLU"]
    P1["Permute\n[B,C,H,W] -> [B,H,W,C]"]

    subgraph BLKS["Swin Blocks (depth=2, 交替 shift)"]
      B0_IN["Block 0 输入\n[B,Hs,Ws,C]"]
      B0_N1["LayerNorm(C)"]
      B0_PAD["可选 Pad 到 window 整除\npad_to_window=True"]
      B0_WA["WindowAttention\nqkv: Linear(C->3C)\nheads=h, head_dim=C/h\nsoftmax(attn)\n+ 相对位置偏置(可选)\nproj: Linear(C->C)"]
      B0_R1["Residual Add #1"]
      B0_N2["LayerNorm(C)"]
      B0_MLP["MLP\nLinear(C->4C) + GELU + Linear(4C->C)"]
      B0_R2["Residual Add #2"]

      B1_IN["Block 1 输入\n[B,Hs,Ws,C]"]
      B1_N1["LayerNorm(C)"]
      B1_SHIFT["Shifted Window\nroll(-4,-4), ws=8\n(若 use_shift_mask=True 则加 mask)"]
      B1_WA["WindowAttention\n(同上，窗口内注意力)"]
      B1_REV["Reverse Shift\nroll(+4,+4) + 去 Pad"]
      B1_R1["Residual Add #1"]
      B1_N2["LayerNorm(C)"]
      B1_MLP["MLP\nLinear(C->4C)+GELU+Linear(4C->C)"]
      B1_R2["Residual Add #2"]
    end

    NOUT["Stage Tail\nLayerNorm(C)"]
    P2["Permute\n[B,H,W,C] -> [B,C,H,W]"]

    OUT3["Stage3 输出 x3\nB x 256 x H/16 x W/16"]
    OUT4["Stage4 输出 x4\nB x 512 x H/32 x W/32"]

    IN3 --> DS
    IN4 --> DS
    DS --> P1 --> B0_IN --> B0_N1 --> B0_PAD --> B0_WA --> B0_R1 --> B0_N2 --> B0_MLP --> B0_R2
    B0_R2 --> B1_IN --> B1_N1 --> B1_SHIFT --> B1_WA --> B1_REV --> B1_R1 --> B1_N2 --> B1_MLP --> B1_R2
    B1_R2 --> NOUT --> P2
    P2 --> OUT3
    P2 --> OUT4
```

## 2) 频域解耦瓶颈（WaveletDecoupledBottleneck）细化图

```mermaid
flowchart TD
    X["输入 x\nB x 512 x H/32 x W/32"]

    DWT["Haar DWT2D\n(奇数尺寸先 replicate pad)\n分解为 ll/lh/hl/hh\n每支: B x 512 x H/64 x W/64"]

    subgraph LOW["低频支路"]
      L1["low_proj\nConv1x1 512->512 (bias=False)\n+ Norm2d(bn/gn)\n+ GELU"]
    end

    subgraph HIGH["高频支路"]
      HC["Concat(lh,hl,hh)\nB x 1536 x H/64 x W/64"]
      HR0["HighFrequencyRefiner 残差输入"]
      HR1["reduce\nConv1x1 1536->512 (bias=False)"]
      HR2["Norm2d + GELU"]
      HR3["dw+pw\nDepthwise Conv3x3 (groups=512)\n+ Pointwise Conv1x1 512->512"]
      HR4["Norm2d + GELU"]
      HR5["expand\nConv1x1 512->1536 (bias=False)"]
      HR6["残差相加\n(与 HR0 相加)"]
      SPLIT["chunk -> lh/hl/hh\n各 B x 512 x H/64 x W/64"]
      HC --> HR0 --> HR1 --> HR2 --> HR3 --> HR4 --> HR5 --> HR6 --> SPLIT
    end

    IDWT["Haar IDWT2D\n重建 rec: B x 512 x H/32 x W/32\n并裁剪回原 out_hw"]
    FUSE["fuse\nConv3x3 512->512 (bias=False)\n+ Norm2d(bn/gn)\n+ GELU"]
    Y["输出 y = x + fuse(rec)\nB x 512 x H/32 x W/32"]

    X --> DWT
    DWT --> L1
    DWT --> HC
    L1 --> IDWT
    SPLIT --> IDWT
    IDWT --> FUSE --> Y
    X --> Y
```

## 说明

- `SwinBlock` 的标准顺序是：`LN -> (Window Attention) -> Residual -> LN -> MLP -> Residual`。
- `depth=2` 时，block0 是非移位窗口，block1 是移位窗口（`shift_size=window_size//2=4`）。
- 频域模块中，高频分支是一个轻量残差卷积细化器（`1x1 reduce -> DWConv+PWConv -> 1x1 expand`）。
