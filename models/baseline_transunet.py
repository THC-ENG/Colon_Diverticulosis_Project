import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSUNET_ROOT = REPO_ROOT / "external_models" / "TransUNet"
if str(TRANSUNET_ROOT) not in sys.path:
    sys.path.append(str(TRANSUNET_ROOT))

from networks.vit_seg_modeling import CONFIGS as TRANSUNET_CONFIGS  # noqa: E402
from networks.vit_seg_modeling import VisionTransformer  # noqa: E402


class _SlashKeyNpz:
    def __init__(self, weights):
        self.weights = weights

    def __getitem__(self, key):
        return self.weights[str(key).replace("\\", "/")]


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        norm_type: str = "bn",
        img_size: int = 256,
        vit_name: str = "R50-ViT-B_16",
        n_skip: int = 3,
        vit_patches_size: int = 16,
        pretrained_path: str = "checkpoints/R50+ViT-B_16.npz",
    ):
        super().__init__()
        if int(in_channels) != 3 or int(num_classes) != 1:
            raise ValueError("TransUNet baseline currently supports RGB input and binary output only.")

        config = TRANSUNET_CONFIGS[str(vit_name)]
        config.n_classes = int(num_classes)
        config.n_skip = int(n_skip)
        if "R50" in str(vit_name):
            grid = int(int(img_size) / int(vit_patches_size))
            config.patches.grid = (grid, grid)

        self.model = VisionTransformer(config, img_size=int(img_size), num_classes=int(num_classes))
        path = REPO_ROOT / pretrained_path
        if not path.exists():
            raise FileNotFoundError(f"TransUNet pretrained checkpoint not found: {path}")
        weights = _SlashKeyNpz(np.load(path))
        self.model.load_from(weights=weights)

    def forward(self, x):
        return self.model(x)
