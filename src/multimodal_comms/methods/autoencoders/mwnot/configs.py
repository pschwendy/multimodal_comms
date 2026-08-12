from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class MWNOTConfig:
    """Configuration for MWNOT.

    M: base WMGM dimension.
    lifting: "legendre" or "linear".
    use_wavelets: if False, tokens are produced from the lifted map directly.
    """

    M: int = 4
    patch_size: int = 5
    poly_order: int = 3
    linear_lift_channels: int = 16
    lifting: Literal["legendre", "linear"] = "legendre"
    use_wavelets: bool = True
    wavelet_levels: int = 3
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.1
    max_levels: int = 16
    sort_nodes: bool = True
    symmetrize_input: bool = True
    normalize_input: bool = True
    preserve_zeros: bool = True
    enforce_p_symmetry: bool = True
    p_loss_weight: float = 1.0
    l_loss_weight: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 0
    train_samples: int = 1024
    val_samples: int = 128
    batch_size: int = 16
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    data_K: int = 3
    min_nodes: int | None = None
    max_nodes: int | None = None
    lambda_l: float = 1.0
