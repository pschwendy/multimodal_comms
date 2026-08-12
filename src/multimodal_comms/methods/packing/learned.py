"""Capacity-matched packet packing: MANY messages, one fixed-size packet.

Why this module exists (and why superpose.py cannot get past ~16 messages):
`superpose.py` binds every message across the FULL latent space,
Z = sum_i z_i @ Q_i with Q_i a (D, D) rotation, so every message spends the
whole packet and messages collide head-on. Unbinding slot j gives
z_j + sum_{i!=j} z_i Q_i Q_j^T, whose crosstalk energy grows like (M-1)
times the signal -- SNR ~ 1/sqrt(M-1). Measured behaviour matches: usable at
M=4, marginal at M=8, noise at M=16. No amount of decoder training fixes
that, because the packet simply is not big enough to hold M full-width
messages: it is an information-capacity wall, not a denoising problem.

The fix is to make each message's footprint SMALL instead of making the
collision quieter. A packet is P real numbers. Give message i a code
c_i in R^d with d ~ P/M, and the packet has room for M of them. Everything
here is built around that:

  1. `PackedBottleneck` -- a trained encoder/decoder head that squeezes a
     message's (K, H) raw latents into d dims and back. MATRYOSHKA-nested
     (Kusupati et al. 2022): one trained module serves EVERY width in
     `ladder` because training randomly truncates the code to a prefix, so
     a single checkpoint gives the whole quality-vs-message-count frontier
     instead of one training run per operating point.

  2. Packers -- how M codes share P numbers. All three sum the messages
     into ONE fixed-size packet; they differ in whether the sum is a
     genuine superposition and in what it costs:

       `RotorPacker`  THE HEADLINE. Slot i's frame is rows [i*d,(i+1)*d) of
                      a keyed dense rotation R of the whole packet, so
                      packet = (concat_i c_i) R. Frames are mutually
                      orthogonal (exact recovery, M = P//d) yet every
                      message is DENSE across all P coordinates -- no
                      number in the packet belongs to any one sender. Plus
                      a private per-slot rotation V_i, which makes another
                      receiver's view of your code Haar-uniform and so
                      information-theoretically empty. Costs one rotation
                      per packet, not M frame builds.
       `FramePacker`  per-slot INDEPENDENT random semi-orthogonal frame
                      A_i (d, P), packet = sum_i c_i A_i. Also genuinely
                      fused. Needs no coordination between senders and
                      still functions when OVERLOADED (M*d > P), which the
                      other two structurally cannot -- paid for in
                      crosstalk.
       `BlockPacker`  disjoint coordinate blocks: plain concatenation, NOT
                      a fusion scheme. Reference only. RotorPacker matches
                      it exactly, which is the point -- density of
                      superposition costs no fidelity whatsoever.

    Disjoint SUBSPACES are not disjoint COORDINATES: that distinction is
    what lets RotorPacker be simultaneously fused, exact, and private.

The one number that governs FramePacker is the load factor

    rho = M * d / P            (fraction of packet capacity requested)

Unbinding gives c_j + sum_{i!=j} c_i A_i A_j^T. For independent Haar frames
E||c A_i A_j^T||^2 = ||c||^2 * d / P, so with unit-RMS codes (||c||^2 = d)
the per-dimension crosstalk variance is (M-1) * d / P ~ rho against unit
signal variance:

    SNR ~ 1 / rho     (rho=1 -> 0 dB, rho=1/2 -> 3 dB, rho=1/4 -> 6 dB)

-- flat in M at fixed rho, which is the whole point: doubling the message
count at half the code width costs nothing in SNR, whereas superpose.py's
full-width binding is rho = M and degrades with every added message.
`crosstalk_std` is that sqrt(rho), and the trainer injects exactly this
noise so one checkpoint covers both packers.
"""

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn


DEFAULT_LADDER = (2560, 1280, 640, 320, 160, 80, 40)
"""Nested code widths. Chosen so P // d is a clean message count at the
standard packet size P = 4 * 2560 = 10240 floats (40 KB fp32): the ladder
reads directly as M = 4, 8, 16, 32, 64, 128, 256 messages per packet."""


def crosstalk_std(n_messages: int, code_dim: int, packet_dim: int) -> float:
    """Per-dimension std of FramePacker crosstalk, for unit-RMS codes.

    Exact in expectation (see module docstring): crosstalk energy is
    (M-1) * ||c||^2 * d / P spread over d dims, and ||c||^2 = d, giving
    variance (M-1) * d / P. Returned as a std so the trainer can add
    matched Gaussian noise without ever materialising a packet.
    """
    return math.sqrt(max(n_messages - 1, 0) * code_dim / packet_dim)


class PackedBottleneck(nn.Module):
    """(K, H) raw latents <-> a d-dim code, with nested (matryoshka) widths.

    down: flatten(K*H) -> width -> code_dim, up: code_dim -> width -> K*H.
    An MLP, not a plain Linear: at the widths that matter here (d << K*H)
    a linear map is a strict rank bottleneck on a latent set that is not a
    linear subspace, and the nonlinearity is where most of the small-d
    quality comes from.

    NESTING: `truncate` keeps the first `d` code dims and zeroes the rest,
    so up() always sees a code_dim-long vector and any prefix width is a
    valid operating point. Training samples d from the ladder, which is
    what makes the prefixes actually informative rather than arbitrary.

    NORMALISATION: codes are scaled to unit RMS over the KEPT dims. This is
    not cosmetic -- it is what makes `crosstalk_std` a fixed function of
    (M, d, P) rather than of whatever scale the encoder happened to learn,
    so the noise the trainer injects matches the noise a real packet
    delivers. The discarded scalar norm is restored on the receive side by
    the up-projection's own learned scale (it never sees anything but
    unit-RMS codes, in training or at runtime).
    """

    def __init__(self, num_latents: int, hidden: int, code_dim: int = 2560,
                 width: int = 4096, slot_dim: int = 0, identity: bool = False,
                 dtype=torch.float32):
        super().__init__()
        # identity=True: TRUE SUPERPOSITION. No dimensional reduction anywhere --
        # the code is the standardised latent itself, so a message spans the
        # entire packet and receivers must separate it from the others' by key
        # alone. Disjoint-subspace packing is unavailable by construction here
        # (a full-width code leaves exactly one slot), which is the point.
        self.identity = identity
        self.num_latents = num_latents
        self.hidden = hidden
        self.code_dim = code_dim
        in_dim = num_latents * hidden
        self.in_dim = in_dim
        # FACTORISED mode (slot_dim > 0), for long-context checkpoints where K
        # is large. A flat MLP over K*H is quadratic in the wrong place: at
        # K=16, H=2560 the first layer alone is 40960 x width (~105M params at
        # width=2560, and it has to be mirrored on the way back up). Projecting
        # each slot H -> slot_dim with SHARED weights first cuts that by
        # H/slot_dim and is the better inductive bias anyway, since the slots
        # are sequence positions rather than arbitrary coordinates: the shared
        # projection says "compress a position", and the joint MLP afterwards
        # says "trade capacity between positions".
        self.slot_dim = slot_dim
        if slot_dim:
            self.slot_down = nn.Linear(hidden, slot_dim, dtype=dtype)
            self.slot_up = nn.Linear(slot_dim, hidden, dtype=dtype)
            joint = num_latents * slot_dim
        else:
            self.slot_down = self.slot_up = None
            joint = in_dim
        self.joint_dim = joint
        # width MUST exceed code_dim: the hidden layer is a rank ceiling on
        # everything the code can carry, so width < code_dim silently caps
        # the widest ladder rungs (an earlier run had width=2048 under
        # code_dim=2560 and every rung scored the same, because none of them
        # could use more than 2048 dims).
        if width < code_dim and not identity:
            raise ValueError(
                f"width {width} < code_dim {code_dim}: the MLP hidden layer "
                f"would cap the code's usable rank below its nominal width.")
        self.down = nn.Sequential(
            nn.Linear(joint, width, dtype=dtype),
            nn.GELU(),
            nn.Linear(width, code_dim, dtype=dtype),
        )
        self.up = nn.Sequential(
            nn.Linear(code_dim, width, dtype=dtype),
            nn.GELU(),
            nn.Linear(width, joint, dtype=dtype),
        )
        # Fixed standardisation, fitted once from data (fit_stats) and held
        # as buffers -- NOT a LayerNorm. Raw hidden states carry a large
        # roughly-constant offset (the "generic language" component that puts
        # unrelated latents at cosine ~0.3, see
        # reports/crypto_autoencoder_security_20260721.md), and the bottleneck
        # should not spend code dims re-encoding a constant. Fixed buffers
        # rather than a learned norm because `decode` has to INVERT the
        # transform exactly to hand the LM latents on its own native scale;
        # a learned affine makes that inverse a moving target during
        # training, which is what lets the decoder drift into ignoring the
        # latent slots altogether.
        self.register_buffer("mu", torch.zeros(in_dim, dtype=dtype))
        self.register_buffer("sigma", torch.ones(in_dim, dtype=dtype))

    @torch.no_grad()
    def fit_stats(self, latents: torch.Tensor):
        """Set mu/sigma from a sample of real (B, K, H) latents.

        Sigma is floored RELATIVE to the median, not just at 1e-3: from a small
        sample over P~1e5 dims, a few dims draw a spuriously tiny std, and an
        absolute floor leaves them under-scaled so their standardised values
        blow up. A relative floor keeps every standardised dim O(1), so the
        code stays genuinely ~unit-RMS (the crosstalk law's calibration)."""
        x = latents.reshape(latents.shape[0], -1).to(self.mu.dtype)
        self.mu.copy_(x.mean(0))
        s = x.std(0)
        self.sigma.copy_(s.clamp_min(max(1e-3, 0.1 * s.median().item())))

    def standardize(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.reshape(latents.shape[0], -1).to(self.mu.dtype)
        return (x - self.mu) / self.sigma

    def _to_joint(self, latents: torch.Tensor) -> torch.Tensor:
        """(B, K, H) -> (B, joint_dim), standardised (and slot-projected)."""
        x = self.standardize(latents)
        if self.slot_dim:
            x = self.slot_down(x.view(-1, self.num_latents, self.hidden))
            x = x.reshape(x.shape[0], -1)
        return x

    def _from_joint(self, x: torch.Tensor) -> torch.Tensor:
        """(B, joint_dim) -> (B, in_dim) in STANDARDISED space."""
        if self.slot_dim:
            x = self.slot_up(x.view(-1, self.num_latents, self.slot_dim))
            x = x.reshape(x.shape[0], -1)
        return x

    def encode(self, latents: torch.Tensor, code_dim: int | None = None
               ) -> torch.Tensor:
        """(B, K, H) -> (B, code_dim) unit-RMS code, zeroed past `code_dim`."""
        if self.identity:
            # The code IS the standardised latent, so decode (code*sigma+mu) is
            # an EXACT round-trip. Do NOT apply truncate's per-sample rsqrt here:
            # standardisation already makes the code ~unit-RMS, and at high P the
            # fitted per-dim sigma is noisy -- a few under-scaled dims inflate the
            # per-sample RMS, and dividing by that RMS (which decode cannot
            # invert) injects wrong-scale latents and collapses reconstruction to
            # chance. (This silently sank the K=96/P=393216 frontier: M=1 sat at
            # the 0.013 prior with MSE in the millions, while K=48 survived only
            # because its RMS stayed near 1.)
            return self.standardize(latents)
        return self.truncate(self.down(self._to_joint(latents)), code_dim)

    def truncate(self, code: torch.Tensor, code_dim: int | None = None
                 ) -> torch.Tensor:
        d = self.code_dim if code_dim is None else code_dim
        kept = code[..., :d]
        # rsqrt(mean(x^2)) rather than /norm: keeps per-dim signal variance
        # at 1 for EVERY d, which is what crosstalk_std is calibrated against.
        kept = kept * torch.rsqrt(kept.pow(2).mean(-1, keepdim=True) + 1e-6)
        if d == code.shape[-1]:
            return kept
        pad = torch.zeros(*code.shape[:-1], code.shape[-1] - d,
                          dtype=kept.dtype, device=kept.device)
        return torch.cat([kept, pad], dim=-1)

    def decode(self, code: torch.Tensor, out_dtype=None) -> torch.Tensor:
        """(B, code_dim) -> (B, K, H) latents ready for the decode prompt.

        Undoes `standardize`, so the LM receives latents on the native scale
        it was pretrained to read."""
        if self.identity:
            x = code[..., :self.in_dim].to(self.mu.dtype) * self.sigma + self.mu
        else:
            x = self._from_joint(self.up(code.to(self.up[0].weight.dtype)))
            x = x * self.sigma + self.mu
        x = x.view(-1, self.num_latents, self.hidden)
        return x if out_dtype is None else x.to(out_dtype)

    def recon_loss(self, latents: torch.Tensor, code: torch.Tensor
                   ) -> torch.Tensor:
        """MSE between up(code) and the true latents, in STANDARDISED space.

        The auxiliary objective that keeps the bottleneck alive. Training on
        the LM's cross-entropy alone is not enough: the decode pass is
        teacher-forced, so the LM can score ~0.44 token accuracy from the
        text prefix with no help from the latent slots at all, and while the
        freshly-initialised bottleneck is emitting noise that is exactly what
        it learns to do. Once it ignores the slots, no gradient reaches the
        bottleneck and the whole ladder flatlines at the prior (observed:
        every rung from d=2560 to d=40 scoring 0.44-0.47). A direct
        reconstruction term does not depend on the LM attending to anything,
        so it cannot collapse this way.
        """
        if self.identity:
            return nn.functional.mse_loss(code[..., :self.in_dim],
                                          self.standardize(latents))
        pred = self._from_joint(self.up(code.to(self.up[0].weight.dtype)))
        return nn.functional.mse_loss(pred, self.standardize(latents))

    def config(self) -> dict:
        return {"num_latents": self.num_latents, "hidden": self.hidden,
                "code_dim": self.code_dim, "width": self.down[0].out_features,
                "slot_dim": self.slot_dim, "identity": self.identity}


# --------------------------------------------------------------------------
# Packers: M codes -> one P-dim packet -> M recovered codes
# --------------------------------------------------------------------------

class BlockPacker:
    """Disjoint coordinate blocks: slot i owns packet[i*d : (i+1)*d].

    NOT A FUSION SCHEME -- this is plain concatenation, kept only as the
    fidelity REFERENCE. Coordinate j of the packet belongs to exactly one
    message and reveals nothing about any other, so the packet is a
    container, not a shared representation. Anything claiming to superpose
    messages has to be scored against it, and RotorPacker (below) matches it
    EXACTLY while being genuinely fused, which is the whole point: density
    of superposition is free, it costs no reconstruction quality at all.

    Zero crosstalk, bit-exact recovery, capacity M = P // d.
    """

    def __init__(self, packet_dim: int, code_dim: int):
        if code_dim > packet_dim:
            raise ValueError(f"code_dim {code_dim} > packet_dim {packet_dim}")
        self.packet_dim = packet_dim
        self.code_dim = code_dim
        self.capacity = packet_dim // code_dim

    def pack(self, codes_by_slot: dict[int, torch.Tensor]) -> torch.Tensor:
        first = next(iter(codes_by_slot.values()))
        packet = torch.zeros(self.packet_dim, dtype=torch.float32,
                             device=first.device)
        for slot, c in codes_by_slot.items():
            if not 0 <= slot < self.capacity:
                raise ValueError(
                    f"slot {slot} outside 0..{self.capacity - 1}: a {self.packet_dim}-dim "
                    f"packet holds exactly {self.capacity} codes of width "
                    f"{self.code_dim}. Reduce code_dim or use FramePacker to overload.")
            packet[slot * self.code_dim:(slot + 1) * self.code_dim] = (
                c[:self.code_dim].to(torch.float32))
        return packet

    def unpack(self, packet: torch.Tensor, slot: int) -> torch.Tensor:
        return packet[slot * self.code_dim:(slot + 1) * self.code_dim]


class FramePacker:
    """Per-slot random semi-orthogonal frame: packet = sum_i c_i A_i.

    A_i is (d, P) with orthonormal rows, drawn deterministically from
    (seed, slot) -- int seed for public multiplexing, dict[slot, secret] for
    private per-receiver keys, matching superpose.py's two-mode convention
    (and its `nonce` discipline: a linear map under a REUSED key falls to a
    known-plaintext linear solve, so pass a fresh nonce per packet whenever
    confidentiality is claimed).

    Unpack is A_j^T applied to the packet: c_j + sum_{i!=j} c_i A_i A_j^T.
    Crosstalk is governed by rho = M*d/P alone (module docstring), so unlike
    superpose.py's full-width rotation this degrades with how much of the
    packet is *requested*, not with how many messages there are -- and it
    keeps working past rho = 1, where BlockPacker simply has no slots left.

    Frames are built by QR on a (P, d) Gaussian: cost is O(P d^2), a few ms
    at P=10240, d<=2560, and they are cached per (slot, nonce).
    """

    def __init__(self, packet_dim: int, code_dim: int,
                 seed: int | dict[int, int] = 1234, nonce: int | None = None,
                 device: str | torch.device = "cpu", cache_limit: int = 512):
        self.packet_dim = packet_dim
        self.code_dim = code_dim
        self.seed = seed
        self.nonce = nonce
        self.device = device
        self.cache_limit = cache_limit
        self._cache: dict[int, torch.Tensor] = {}

    def _slot_seed(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(
                    f"slot {slot} has no enrolled private secret in this packer "
                    f"(private mode: only slots {sorted(self.seed)} are known here)")
            base = self.seed[slot]
        else:
            base = (self.seed * 1_000_003 + slot * 7_919) % (2**31 - 1)
        if self.nonce is None:
            return base
        digest = hashlib.blake2b(f"{base}:{self.nonce}".encode(), digest_size=8,
                                 person=b"framepk1").digest()
        return int.from_bytes(digest, "big") % (2**63 - 1)

    def frame(self, slot: int) -> torch.Tensor:
        """(d, P) with orthonormal rows."""
        if slot in self._cache:
            return self._cache[slot]
        gen = torch.Generator().manual_seed(self._slot_seed(slot))
        g = torch.randn(self.packet_dim, self.code_dim, generator=gen)
        q, r = torch.linalg.qr(g)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        a = q.T.contiguous().to(self.device)
        if len(self._cache) >= self.cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[slot] = a
        return a

    def pack(self, codes_by_slot: dict[int, torch.Tensor]) -> torch.Tensor:
        packet = None
        for slot, c in codes_by_slot.items():
            contrib = c[:self.code_dim].to(torch.float32) @ self.frame(slot)
            packet = contrib if packet is None else packet + contrib
        return packet

    def unpack(self, packet: torch.Tensor, slot: int) -> torch.Tensor:
        return packet.to(torch.float32) @ self.frame(slot).T


class ButterflyRotation:
    """Fast, keyed, EXACTLY orthogonal P x P transform -- without the matrix.

    RotorPacker needs a dense rotation of the whole packet. Materialising
    one at P=10240 means a 100M-entry matrix (400 MB fp32) and a QR that
    takes ~a minute, which is unusable when a fresh nonce should give a
    fresh rotation per packet. Instead compose cheap exactly-orthogonal
    pieces, the standard fast-random-rotation construction:

        R = prod over rounds of  [ block-diagonal Haar blocks ; permutation ]

    Each factor is orthogonal (block-diagonal-of-orthogonal, and a
    permutation), so R is orthogonal to machine precision by construction --
    no approximation anywhere. The permutation between rounds is what makes
    it MIX: after `rounds` rounds every input coordinate has reached every
    output coordinate (block size b, so reach is b**rounds >= P for the
    defaults). Cost is O(P*b) per apply instead of O(P^2), and the stored
    key material is rounds*(P/b)*b^2 floats instead of P^2 -- 3*20*512^2 =
    16M vs 105M at the defaults, and it builds in well under a second.

    This is a diffusion primitive, not a cipher on its own; it carries the
    shared packet layout, and confidentiality between receivers comes from
    RotorPacker's per-slot private rotations.
    """

    def __init__(self, dim: int, seed: int = 1234, block: int = 512,
                 rounds: int = 3, device: str | torch.device = "cpu"):
        if dim % block != 0:
            block = math.gcd(dim, block) or 1
        self.dim = dim
        self.block = block
        self.rounds = rounds
        self.device = device
        gen = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
        n_blocks = dim // block
        self._blocks, self._perms = [], []
        for _ in range(rounds):
            g = torch.randn(n_blocks, block, block, generator=gen)
            q, r = torch.linalg.qr(g)
            q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
            self._blocks.append(q.to(device))
            self._perms.append(torch.randperm(dim, generator=gen).to(device))

    def _to(self, device):
        """Follow the data's device. Callers legitimately pack CPU codes in
        one place and GPU codes in another (the evaluation harness does
        both), and silently raising a device-mismatch there would be a
        needless trap."""
        if str(device) == str(self.device):
            return
        self._blocks = [b.to(device) for b in self._blocks]
        self._perms = [p.to(device) for p in self._perms]
        self.device = device

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., P) -> (..., P), multiplication by R."""
        self._to(x.device)
        shape = x.shape
        y = x.reshape(-1, self.dim).to(torch.float32)
        for q, perm in zip(self._blocks, self._perms):
            y = torch.einsum("nbi,bij->nbj", y.view(-1, self.dim // self.block,
                                                    self.block), q)
            y = y.reshape(-1, self.dim)[:, perm]
        return y.reshape(shape)

    def apply_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplication by R^T -- exact inverse of `apply`."""
        self._to(x.device)
        shape = x.shape
        y = x.reshape(-1, self.dim).to(torch.float32)
        for q, perm in zip(reversed(self._blocks), reversed(self._perms)):
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(self.dim, device=perm.device)
            y = y[:, inv]
            y = torch.einsum("nbj,bij->nbi", y.view(-1, self.dim // self.block,
                                                    self.block), q)
            y = y.reshape(-1, self.dim)
        return y.reshape(shape)


class RotorPacker:
    """Genuine dense superposition at ZERO crosstalk: the headline scheme.

    The observation this is built on is that disjoint SUBSPACES are not
    disjoint COORDINATES. Take a dense keyed rotation R of the whole packet
    and give slot i the frame A_i = S_i R, i.e. rows [i*d, (i+1)*d) of R:

        packet = sum_i c_i A_i = ( concat_i c_i ) R

    Every A_i has orthonormal rows and A_i A_j^T = 0 for i != j, so recovery
    is EXACT at any load up to M = P // d -- and yet each c_i A_i is a dense
    vector touching all P coordinates, so no number in the packet belongs to
    any one message. Read any coordinate and you are reading a mixture of
    all M messages; the packet is one representation that means a different
    thing under each receiver's key. That is superposition in the sense that
    matters, and it costs exactly nothing in fidelity: RotorPacker is
    bit-identical to BlockPacker composed with a rotation, so it sits ON the
    concatenation ceiling rather than below it.

    Because pack factors as "concatenate, then rotate once", the cost is ONE
    O(P log P)-ish ButterflyRotation per packet -- not M frame builds -- so
    it is by far the cheapest packer here as well as the most accurate.

    CONFIDENTIALITY (`private=True`, the default). R is necessarily shared
    (it is the layout every receiver inverts), so R alone would let any
    receiver slice out any slot's block. Each slot therefore also gets its
    own secret d x d Haar rotation V_i applied to the code BEFORE placement:

        packet = ( concat_i  c_i V_i ) R

    Receiver i undoes R, takes its block, and applies V_i^T -- exact. An
    insider holding R and its own V_j sees c_i V_i for every other slot,
    and by right-invariance of the Haar measure c_i V_i is uniform on the
    sphere of radius ||c_i|| whatever c_i was, so its DIRECTION carries zero
    mutual information (the Theorem-1 argument of
    reports/crypto_provable_security_20260722.md). Codes are unit-RMS, so
    the norm is a known constant and leaks nothing either. Net: exact
    reconstruction and provably zero content leakage, simultaneously.

    The Gram-matrix break that sank the old scheme (superpose.py's
    _derive_row_seed) cannot recur here: a slot holds ONE code vector, not K
    rows under a shared key, so there are no pairwise inner products to
    survive the rotation. Keep it that way -- one vector per slot.
    """

    def __init__(self, packet_dim: int, code_dim: int,
                 seed: int | dict[int, int] = 1234, nonce: int | None = None,
                 layout_seed: int = 1234, private: bool = True,
                 device: str | torch.device = "cpu", rounds: int = 3):
        if code_dim > packet_dim:
            raise ValueError(f"code_dim {code_dim} > packet_dim {packet_dim}")
        self.packet_dim = packet_dim
        self.code_dim = code_dim
        self.capacity = packet_dim // code_dim
        self.seed = seed
        self.nonce = nonce
        self.private = private
        self.device = device
        layout = layout_seed if nonce is None else int.from_bytes(
            hashlib.blake2b(f"{layout_seed}:{nonce}".encode(), digest_size=8,
                            person=b"rotorlay").digest(), "big") % (2**63 - 1)
        self.rotation = ButterflyRotation(packet_dim, seed=layout, rounds=rounds,
                                          device=device)
        self._v: dict[int, torch.Tensor] = {}

    def _slot_seed(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(
                    f"slot {slot} has no enrolled private secret in this packer "
                    f"(private mode: only slots {sorted(self.seed)} are known here)")
            base = self.seed[slot]
        else:
            base = (self.seed * 1_000_003 + slot * 7_919) % (2**31 - 1)
        if self.nonce is None:
            return base
        return int.from_bytes(
            hashlib.blake2b(f"{base}:{self.nonce}".encode(), digest_size=8,
                            person=b"rotorslt").digest(), "big") % (2**63 - 1)

    def code_rotation(self, slot: int) -> torch.Tensor | None:
        """Slot's private (d, d) Haar rotation V_i, or None if public mode."""
        if not self.private:
            return None
        if slot not in self._v:
            gen = torch.Generator().manual_seed(self._slot_seed(slot))
            g = torch.randn(self.code_dim, self.code_dim, generator=gen)
            q, r = torch.linalg.qr(g)
            self._v[slot] = (q * torch.sign(torch.diagonal(r)).unsqueeze(0)
                             ).to(self.device)
        return self._v[slot]

    def _check(self, slot: int):
        if not 0 <= slot < self.capacity:
            raise ValueError(
                f"slot {slot} outside 0..{self.capacity - 1}: a {self.packet_dim}-dim "
                f"packet holds {self.capacity} codes of width {self.code_dim} at zero "
                f"crosstalk. Narrow the code, or use FramePacker to overload past it.")

    def pack(self, codes_by_slot: dict[int, torch.Tensor]) -> torch.Tensor:
        first = next(iter(codes_by_slot.values()))
        flat = torch.zeros(self.packet_dim, dtype=torch.float32, device=first.device)
        for slot, c in codes_by_slot.items():
            self._check(slot)
            c = c[:self.code_dim].to(torch.float32)
            v = self.code_rotation(slot)
            if v is not None:
                c = c @ v.to(c.device)
            flat[slot * self.code_dim:(slot + 1) * self.code_dim] = c
        return self.rotation.apply(flat)

    def unpack(self, packet: torch.Tensor, slot: int) -> torch.Tensor:
        self._check(slot)
        flat = self.rotation.apply_inverse(packet.to(torch.float32))
        c = flat[slot * self.code_dim:(slot + 1) * self.code_dim]
        v = self.code_rotation(slot)
        return c if v is None else c @ v.to(c.device).T


def build_packer(kind: str, packet_dim: int, code_dim: int, **kwargs):
    if kind == "block":
        for k in ("seed", "nonce", "device", "layout_seed", "private", "rounds"):
            kwargs.pop(k, None)
        return BlockPacker(packet_dim, code_dim)
    if kind == "rotor":
        kwargs.pop("cache_limit", None)
        return RotorPacker(packet_dim, code_dim, **kwargs)
    if kind == "frame":
        for k in ("layout_seed", "private", "rounds"):
            kwargs.pop(k, None)
        return FramePacker(packet_dim, code_dim, **kwargs)
    raise ValueError(
        f"Unknown packer kind: {kind!r} (expected 'rotor', 'frame' or 'block')")


# --------------------------------------------------------------------------
# Leakage: what one receiver learns about another receiver's message
# --------------------------------------------------------------------------

def frame_leakage_bound(n_messages: int, code_dim: int, packet_dim: int) -> dict:
    """The measurable leakage term for FramePacker, as a closed form.

    Receiver j's entire view is v_j = c_j + sum_{i!=j} c_i (A_i A_j^T). For
    an independent Haar frame A_i, c_i A_i is uniform on the sphere of
    radius ||c_i|| in R^P regardless of c_i's direction (rotational
    invariance of the Haar measure -- the same argument as Theorem 1-3 of
    reports/crypto_provable_security_20260722.md, which is stated for a full
    rotation but holds verbatim for a semi-orthogonal frame because its rows
    span a Haar-random d-subspace). Projecting a uniform-direction vector by
    the fixed A_j^T leaves the direction uniform in R^d. So the DIRECTION of
    every other message is information-theoretically absent from v_j:

        I(direction(c_i) ; v_j) = 0   exactly, for all i != j

    and by the data-processing inequality no decoder, probe, or amount of
    compute recovers it. What survives is the NORM profile -- but codes here
    are unit-RMS by construction (PackedBottleneck.truncate), so every
    message's norm is the SAME known constant sqrt(d) and even that channel
    carries zero bits. The residual leakage is then purely the crosstalk
    ENERGY a receiver observes, which reveals only how many slots are
    occupied:

        E||v_j - c_j||^2 / ||c_j||^2 = (M-1) * d / P  =  rho - d/P

    That is the number this function returns as `energy_leak`, and it is a
    function of (M, d, P) only -- never of message content. It is what
    should be quoted as "advantage is bounded by a measurable
    representational leakage term": the bound is a load statistic, and the
    content term is exactly zero.

    CAVEAT, stated because it is the same shape as the bug that broke the
    old scheme (see _derive_row_seed in superpose.py): this is a claim about
    ONE code vector per slot. It holds here only because a message is a
    SINGLE d-dim code, so there is no K x K Gram matrix to leak. Do not
    re-introduce multiple rows per slot under a shared frame without
    redoing the analysis.
    """
    rho = n_messages * code_dim / packet_dim
    return {
        "n_messages": n_messages, "code_dim": code_dim, "packet_dim": packet_dim,
        "rho": rho,
        "energy_leak": (n_messages - 1) * code_dim / packet_dim,
        "direction_leak_bits": 0.0,
        "norm_leak_bits": 0.0,
        "snr_db": 10 * math.log10(1.0 / max(crosstalk_std(n_messages, code_dim,
                                                          packet_dim) ** 2, 1e-12)),
    }


def measure_leakage(packer, codes_by_slot: dict[int, torch.Tensor],
                    victim: int, attacker: int) -> dict:
    """Empirical check of `frame_leakage_bound` on real codes.

    Reports the cosine between the victim's true code and what the attacker
    actually observes (its own unpack output), against the matched chance
    floor obtained by re-running with the victim's code replaced by an
    independent random one. `advantage` is the difference; the bound above
    predicts 0 up to sampling noise.
    """
    packet = packer.pack(codes_by_slot)
    view = packer.unpack(packet, attacker)
    truth = codes_by_slot[victim][:packer.code_dim].to(torch.float32)

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

    sham = dict(codes_by_slot)
    r = torch.randn_like(truth)
    sham[victim] = r * torch.rsqrt(r.pow(2).mean() + 1e-6)
    view_sham = packer.unpack(packer.pack(sham), attacker)

    observed = cos(view, truth)
    chance = cos(view_sham, truth)
    return {"observed_cos": observed, "chance_cos": chance,
            "advantage": observed - chance}


# --------------------------------------------------------------------------
# Runtime codec: text <-> code, wrapping the trained LM + bottleneck
# --------------------------------------------------------------------------

class PackedCodec:
    """text -> d-dim code -> text, using a pretrain_packed.py checkpoint.

    Deliberately mirrors superpose.LatentCodec's encode chat template,
    latent-position rule and RECONSTRUCT decode prompt, so a packed
    checkpoint stays drop-in wherever a LatentCodec is expected; the only
    addition is the PackedBottleneck between the LM's latents and the wire.

    `code_dim=None` uses the full trained width; pass any prefix width from
    the training ladder to move along the message-count frontier (the
    nesting is what makes every prefix a valid operating point).
    """

    def __init__(self, model_path: str, code_dim: int | None = None,
                 device: str | None = None, max_new_tokens: int = 256,
                 max_len: int = 384):
        self.model_path = model_path
        self.code_dim = code_dim
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_len = max_len
        self._model = None
        self._tok = None
        self._bn = None
        self.num_latents = None

    def _load(self):
        if self._model is not None:
            return
        import json as _json
        import os as _os
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = _json.load(open(_os.path.join(self.model_path, "packed_config.json")))
        self.num_latents = cfg["num_latents"]
        self.packet_dim = cfg["packet_dim"]
        self.ladder = cfg["ladder"]
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16).to(self.device).eval()
        ck = torch.load(_os.path.join(self.model_path, "bottleneck.pt"),
                        map_location=self.device, weights_only=False)
        self._bn = PackedBottleneck(**ck["config"]).to(self.device)
        self._bn.load_state_dict(ck["state_dict"])
        self._bn.eval()
        if self.code_dim is None:
            self.code_dim = self._bn.code_dim

    def latents(self, texts: list[str]) -> torch.Tensor:
        """(B, K, H) raw LM latents -- the pre-bottleneck representation."""
        self._load()
        enc = [f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
               for t in texts]
        tokd = self._tok(enc, return_tensors="pt", padding=True, truncation=True,
                         max_length=self.max_len).to(self.device)
        last = self._model.config.num_hidden_layers - 1
        with torch.no_grad():
            out = self._model(tokd["input_ids"],
                              attention_mask=tokd["attention_mask"],
                              output_hidden_states=True)
        hid = out.hidden_states[last]
        lens = tokd["attention_mask"].sum(dim=1).tolist()
        rows = []
        for b, n in enumerate(lens):
            step = n / self.num_latents
            idx = ([n - 1] if self.num_latents == 1 else
                   [min(int(i * step + step / 2), n - 1)
                    for i in range(self.num_latents)])
            rows.append(hid[b, idx, :])
        return torch.stack(rows)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """(B, code_dim) unit-RMS codes -- what actually goes into a packet."""
        self._load()
        with torch.no_grad():
            return self._bn.encode(self.latents(texts).float(), self.code_dim)

    def decode(self, code: torch.Tensor) -> str | None:
        """One (code_dim,) code -> free-running reconstructed text."""
        self._load()
        with torch.no_grad():
            z = self._bn.decode(code.unsqueeze(0).to(self.device),
                                out_dtype=torch.bfloat16)[0]
        tok, model = self._tok, self._model
        prompt = "<|im_start|>user\n" + "".join(
            f"<|L{i}|>" for i in range(self.num_latents)
        ) + "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(prompt, return_tensors="pt",
                         add_special_tokens=False).to(self.device)
        emb = model.get_input_embeddings()
        embeds = emb(ids)
        for i in range(self.num_latents):
            li = tok.convert_tokens_to_ids(f"<|L{i}|>")
            for p in (ids[0] == li).nonzero(as_tuple=True)[0]:
                embeds[0, p] = z[i]
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        gen, past, cur = [], None, embeds
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                out = model(inputs_embeds=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[0, -1, :].argmax().item()
                if nxt == eos:
                    break
                gen.append(nxt)
                cur = emb(torch.tensor([[nxt]], device=self.device))
        txt = tok.decode(gen, skip_special_tokens=True).strip()
        return txt or None


# --------------------------------------------------------------------------
# Wire format: the packet is BYTES, not floats
# --------------------------------------------------------------------------

def quantize_packet(packet: torch.Tensor, bits: int = 8, group: int = 128
                    ) -> tuple[torch.Tensor, dict]:
    """Uniform absmax quantisation of a packet, with per-group scales.

    Message count per packet is set by the packet's BYTE budget, not its
    float count, so this is a direct multiplier on capacity: fp32 -> int8 is
    4x more messages in the same 40 KB, int4 is 8x.

    Fusion makes this unusually cheap here. A RotorPacker packet is a dense
    rotation of concatenated unit-RMS codes, so by rotational mixing every
    packet coordinate is close to N(0, 1) -- no heavy tails, no outlier
    dimensions, all coordinates on the same scale. That is the regime
    uniform quantisation is best in, and it is the opposite of the raw
    latents, where a single per-tensor scale collapses because a few outlier
    dims eat the whole dynamic range (see [[token-decomposed-latents]]:
    4-bit per-latent absmax scored F1 0.067, and only per-group-64 scales
    rescued it). Superposition Gaussianises the wire format for free.

    Group scales are stored fp16 and counted in `bytes`, so the reported
    cost is honest rather than quoting the payload alone.
    """
    if bits >= 32:
        return packet.clone(), {"bits": 32, "bytes": packet.numel() * 4}
    n = packet.numel()
    group = min(group, n)
    pad = (-n) % group
    x = torch.cat([packet, packet.new_zeros(pad)]) if pad else packet
    x = x.view(-1, group)
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    qmax = 2 ** (bits - 1) - 1
    q = torch.round(x / scale * qmax).clamp(-qmax - 1, qmax)
    deq = (q / qmax * scale.half().float()).reshape(-1)[:n]
    n_groups = x.shape[0]
    return deq, {"bits": bits,
                 "bytes": n * bits / 8 + n_groups * 2,
                 "group": group}


def packet_capacity(packet_bytes: int, code_dim: int, bits: int = 8) -> int:
    """How many messages fit in a byte budget at a given code width."""
    return int(packet_bytes * 8 / bits) // code_dim
