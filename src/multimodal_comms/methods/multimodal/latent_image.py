"""Image messages for the packed-packet channel.

The packers in `packing.py` never look at where a code came from -- they
move unit-RMS d-dim vectors around. So adding a modality means adding a
BOTTLENECK, not touching the fusion at all, and a single packet can then
carry text and image messages side by side under the same keys, the same
capacity law M = P/d, and the same leakage bound.

Pipeline, mirroring the text side term for term:

    text :  message -> LM latents (K,H) -> PackedBottleneck  -> d-dim code
    image:  256x256 -> SD-VAE latent (4,32,32) -> ImageBottleneck -> d-dim code

The SD-VAE (stabilityai/sd-vae-ft-ema) is the fixed, pretrained front end;
it already takes 196608 pixel values to 4096 latent values at ~24 dB, and
that 24 dB is the ceiling for everything downstream -- quote reconstruction
numbers against it, not against the original image, or the VAE's own loss
gets charged to the channel.

`ImageBottleneck` is convolutional rather than the text side's MLP because
the VAE latent is spatially structured: at d=80 a flat linear map has to
choose 80 global directions over a 32x32 grid, while a conv stack spends
its budget on a coarse spatial layout, which is what survives compression
at these rates and what makes the difference between "blurry version of
the scene" and "unrelated texture".
"""

import torch
import torch.nn as nn


VAE_LATENT_SHAPE = (4, 32, 32)   # sd-vae-ft-ema at 256x256 input
VAE_SCALE = 0.18215              # standard SD latent scaling factor


class ImageBottleneck(nn.Module):
    """(4,32,32) VAE latent <-> a d-dim matryoshka-nested unit-RMS code.

    Same contract as packing.PackedBottleneck -- `encode(x, d)` returns a
    code whose first d dims are unit-RMS and whose tail is zero, `decode`
    inverts it -- so both modalities' codes are interchangeable inside a
    packet and the crosstalk calibration (packing.crosstalk_std) applies
    unchanged.
    """

    def __init__(self, code_dim: int = 2560, ch: int = 128,
                 latent_shape=VAE_LATENT_SHAPE):
        super().__init__()
        c, h, w = latent_shape
        self.latent_shape = latent_shape
        self.code_dim = code_dim
        # Only TWO stride-2 stages (32 -> 8), not three. An earlier version
        # went 32 -> 4, and it cost ~6 dB at EVERY code width: the spatial
        # collapse to 4x4 destroyed detail that no d could buy back, so the
        # whole quality-vs-width curve was pinned flat by the architecture
        # rather than by the bottleneck. The rule is that the conv stack must
        # not itself be the tightest bottleneck -- the code width has to be.
        self.enc = nn.Sequential(
            nn.Conv2d(c, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 3, 1, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 3, 2, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
        )
        feat = ch * 2 * (h // 4) * (w // 4)
        self.to_code = nn.Linear(feat, code_dim)
        self.from_code = nn.Linear(code_dim, feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch * 2, 4, 2, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, ch, 3, 1, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, c, 3, 1, 1),
        )
        self._feat_shape = (ch * 2, h // 4, w // 4)

    def truncate(self, code: torch.Tensor, code_dim: int | None = None):
        d = self.code_dim if code_dim is None else code_dim
        kept = code[..., :d]
        kept = kept * torch.rsqrt(kept.pow(2).mean(-1, keepdim=True) + 1e-6)
        if d == code.shape[-1]:
            return kept
        pad = torch.zeros(*code.shape[:-1], code.shape[-1] - d,
                          dtype=kept.dtype, device=kept.device)
        return torch.cat([kept, pad], dim=-1)

    def encode(self, latent: torch.Tensor, code_dim: int | None = None):
        """(B,4,32,32) scaled VAE latent -> (B, code_dim) code."""
        h = self.enc(latent)
        return self.truncate(self.to_code(h.flatten(1)), code_dim)

    def decode(self, code: torch.Tensor):
        """(B, code_dim) -> (B,4,32,32) reconstructed VAE latent."""
        h = self.from_code(code).view(-1, *self._feat_shape)
        return self.dec(h)

    def config(self) -> dict:
        return {"code_dim": self.code_dim,
                "ch": self._feat_shape[0] // 2,
                "latent_shape": tuple(self.latent_shape)}


class ImageCodec:
    """image tensor <-> d-dim code, wrapping the frozen SD-VAE + bottleneck.

    Deliberately the image-side twin of packing.PackedCodec: same
    `encode(...) -> code` / `decode(code) -> message` surface, so the
    evaluation and packing code paths do not branch on modality.
    """

    def __init__(self, bottleneck_path: str, vae_name: str = "stabilityai/sd-vae-ft-ema",
                 code_dim: int | None = None, device: str = "cuda"):
        self.device = device
        self.bottleneck_path = bottleneck_path
        self.vae_name = vae_name
        self.code_dim = code_dim
        self._vae = None
        self._bn = None

    def _load(self):
        if self._vae is not None:
            return
        from diffusers import AutoencoderKL

        self._vae = AutoencoderKL.from_pretrained(self.vae_name).to(self.device).eval()
        for p in self._vae.parameters():
            p.requires_grad_(False)
        ck = torch.load(self.bottleneck_path, map_location=self.device,
                        weights_only=False)
        self._bn = ImageBottleneck(**ck["config"]).to(self.device)
        self._bn.load_state_dict(ck["state_dict"])
        self._bn.eval()
        if self.code_dim is None:
            self.code_dim = self._bn.code_dim

    @torch.no_grad()
    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,256,256) in [-1,1] -> (B,4,32,32) scaled latent."""
        self._load()
        return self._vae.encode(images.to(self.device)).latent_dist.mean * VAE_SCALE

    @torch.no_grad()
    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """(B,4,32,32) scaled latent -> (B,3,256,256) image in [-1,1]."""
        self._load()
        return self._vae.decode(latent.to(self.device) / VAE_SCALE).sample.clamp(-1, 1)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self._load()
        return self._bn.encode(self.vae_encode(images), self.code_dim)

    @torch.no_grad()
    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """(B, code_dim) -> (B,3,256,256) reconstructed image in [-1,1]."""
        self._load()
        if code.dim() == 1:
            code = code.unsqueeze(0)
        return self.vae_decode(self._bn.decode(code.to(self.device)))
