"""Superposed latent packets: many messages, one fixed-size packet.

Extends the continuous-latent autoencoder (see AutoencoderCompressor in
channel.py) to a broadcast setting. Each message is encoded to K latent
vectors, bound with a slot-specific orthogonal key matrix, and summed with
every other message into a single packet the size of ONE message's latents:

    Z = sum_i  z_i @ Q_i          # (K, D) regardless of message count

A receiver holding key Q_j unbinds its slot:

    Z @ Q_j^T = z_j + crosstalk   # crosstalk: other messages under random
                                  # rotations, ~zero-mean noise

and the (crosstalk-robust, fine-tuned) decoder LM reconstructs the text.

Everything that must be numerically identical between training
(training.programs.pretrain_superpose) and runtime (SuperposeCompressor, the
multiplex runner) lives here: key generation, bind/unbind, and packet
serialization all happen in float32 through these functions only.

Two distinct keying modes live in this file -- do not mix up which one a
use case needs:

  PUBLIC / multiplexing (seed: int, the default everywhere above):
      key(slot) = f(shared_seed, PUBLIC slot index). Everyone who knows the
      algorithm and the shared seed can compute EVERY slot's key -- this is
      fine (even desirable) for SuperposeCompressor's actual job, which is
      bandwidth compression of a discussion every agent is meant to read in
      full. It provides zero confidentiality: it is a multiplexing scheme,
      not a cipher.

  PRIVATE / broadcast encryption (seed: dict[int, int], via
      mint_receiver_secrets + SecureBroadcastCodec/SecureReceiverCodec
      below): key(slot) = the slot's own independently-minted secret, never
      derived from slot index or any other slot's secret. A receiver
      object built from a single (slot, secret) pair has no key material
      for any other slot -- not "shouldn't", structurally KeyErrors if
      asked. This is the mode for "N different messages, one packet, each
      receiver decodes only its own" -- use it, not the int-seed default,
      whenever that property actually needs to hold.

IMPORTANT caveat on PRIVATE mode by itself: OrthogonalKeyring/
RandomSubspaceKeyring/FeistelKeyring's bind is a LINEAR map z -> z @ Q (or
an affine/near-linear variant). A linear cipher used with a FIXED,
REUSED key is broken by a known-plaintext attack: an attacker who
observes D (plaintext, ciphertext) pairs bound under the SAME Q can solve
Q = Z^-1 C exactly via one linear solve, then decrypt every other message
ever bound under that key, past or future -- entropy of the secret
(2**63) does not help once the key itself has been reconstructed this
way. mint_receiver_secrets alone does not defend against this if the same
secret's derived key is reused across many packets, which is exactly what
happens if you pass the same secrets_by_slot to SecureBroadcastCodec
every time.

The fix (also below): a `nonce` per packet. `nonce=True` (the
SecureBroadcastCodec/SecureReceiverCodec default) mints a fresh random
nonce per packet and mixes it into each slot's derived key via a hash
(`_derive_nonced_seed`), so every packet's key is used exactly once --
there is no way to accumulate matching-key pairs across packets, so the
known-plaintext linear-solve attack above has nothing to accumulate
against. The per-slot SECRET stays constant (that's the receiver's
long-term credential); only the DERIVED KEY changes every packet, the
same relationship a stream cipher's (key, nonce) -> keystream has to a
long-term symmetric key. Cost: with mode="qr", each packet now costs a
fresh (D, D) QR decomposition per slot (~seconds at D=2560) instead of a
cached one -- use mode="sign" (O(D) key generation) if per-packet
freshness at high packet rates matters more than QR's stronger crosstalk
whitening.
"""

import base64
import hashlib
import json
import os
from typing import Any

import numpy as np


def _derive_nonced_seed(secret: int, nonce: int) -> int:
    """secret, nonce -> a fresh derived seed, via a cryptographic hash (not
    a torch.Generator or anything reversible) -- this is the (key, nonce)
    -> per-use-key step that turns a reused long-term secret into a
    single-use key per packet (see module docstring's known-plaintext
    caveat). blake2b, not the builtin hash(): the latter is randomized
    per-process (PYTHONHASHSEED) and not cryptographic, so it would be
    neither deterministic across sender/receiver nor safe here.
    """
    digest = hashlib.blake2b(f"{secret}:{nonce}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


#: Keyring modes whose bind is a single shared LINEAR map over all K latent
#: rows, and which therefore both (a) need per-row keys to avoid leaking the
#: plaintext Gram matrix and (b) implement `row_keys`. FeistelKeyring is
#: excluded because its bind is nonlinear, so the exact
#: (zQ)(zQ)^T = z z^T identity does not hold for it -- that is NOT a claim
#: that Feistel is safe here, only that this specific attack and this
#: specific fix do not transfer to it unanalyzed (and Feistel already fails
#: multi-party reconstruction, see its class docstring).
ROW_KEY_MODES = ("qr", "qr_shake", "sign")


def _derive_row_seed(base_seed: int, row: int) -> int:
    """base_seed, row index -> an independent per-ROW key seed.

    Domain-separated from _derive_nonced_seed (different suffix tag) so a
    (secret, nonce) pair and a (slot_seed, row) pair can never collide onto
    the same derived key material.

    WHY THIS EXISTS (reports/crypto_provable_security_20260722.md Sec 3a):
    a message is K latent vectors, a (K, D) matrix. Binding it as `z @ Q`
    with ONE Q for all K rows makes the whole K x K Gram matrix
    z z^T = (zQ)(zQ)^T an INVARIANT of the cipher -- an adversary reads
    every pairwise inner product (and every row norm) straight off the
    ciphertext, with no known plaintexts and no key reuse. That is a total
    IND-CPA break: challenge with z0 = K orthogonal rows vs z1 = K identical
    rows (identical row norms, so an equal-norm-restricted game admits the
    pair) and the ciphertext's off-diagonal Gram entries give the answer
    with probability 1 (verified: 200/200 in TestGramLeakage).

    Per-ROW independent keys fix it: <z_k Q^(k), z_l Q^(l)> =
    z_k (Q^(k) Q^(l)T) z_l^T, and for independent Haar Q^(k), Q^(l) the
    relative rotation Q^(k) Q^(l)T is itself Haar and independent of the
    plaintexts, so that quantity's distribution depends only on ||z_k||,
    ||z_l|| -- never on the true inner product.
    """
    digest = hashlib.blake2b(
        f"{base_seed}:{row}".encode(), digest_size=8, person=b"rowkey-v1"
    ).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def _shake_gaussian_matrix(seed: int, rows: int, cols: int):
    """seed -> (rows, cols) float32 tensor of iid-standard-Gaussian entries,
    derived entirely from SHAKE256 (a NIST-standardized XOF) rather than
    seeding a non-cryptographic PRNG (torch.Generator: Mersenne Twister on
    CPU, Philox on CUDA). See reports/crypto_provable_security_20260722.md
    Sec 6-7: OrthogonalKeyring's Haar-distribution guarantee (QR of iid
    Gaussians) holds regardless of which generator produced the Gaussians,
    but the *unpredictability* of Q to an adversary without the seed is a
    separate computational assumption -- this generator grounds that
    assumption in SHAKE256's security instead of MT19937/Philox's
    undocumented, un-reduced opacity. Used by OrthogonalKeyring's
    mode="qr_shake" (an alternate to the default "qr", not a replacement --
    changing the default would break bit-compatibility with checkpoints
    already trained against "qr"'s crosstalk statistics; the two produce
    the same DISTRIBUTION, just from different entropy sources, so a
    checkpoint generalizes across them for free, see TestShakeKeyring).
    """
    n = rows * cols
    n_pairs = (n + 1) // 2
    material = seed.to_bytes(8, "big", signed=False) + b"orthkey-shake-v1"
    stream = hashlib.shake_256(material).digest(n_pairs * 2 * 8)
    raw = np.frombuffer(stream, dtype=np.uint64)
    # 8 random bytes -> uint64 -> U(0,1); +0.5 keeps both endpoints open
    # (log(0) in Box-Muller below would otherwise be a rare but real crash).
    u = (raw.astype(np.float64) + 0.5) / 2.0**64
    u1, u2 = u[0::2], u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    g = np.empty(n_pairs * 2, dtype=np.float64)
    g[0::2] = r * np.cos(theta)
    g[1::2] = r * np.sin(theta)
    g = g[:n].astype(np.float32).reshape(rows, cols).copy()
    import torch

    return torch.from_numpy(g)


class OrthogonalKeyring:
    """Deterministic per-slot orthogonal key matrices.

    Keys are regenerated from (seed, slot) on any machine -- nothing is
    stored on disk, so sender and receivers agree by construction.

    mode="qr":   full random orthogonal (D, D) via sign-canonicalized QR of
                 a seeded Gaussian (torch.Generator: MT19937/Philox). Best
                 crosstalk whitening; ~D^2 floats per cached slot.
    mode="qr_shake": identical Haar-distributed construction (same QR
                 argument, see reports/crypto_provable_security_20260722.md
                 Sec 7) but the Gaussian entries come from SHAKE256 (a
                 NIST-standardized XOF) instead of torch.Generator -- grounds
                 the "Q unpredictable without the secret" assumption in a
                 better-studied primitive. Same distribution as "qr", not
                 bit-identical to it; a checkpoint trained under "qr"
                 generalizes to it for free (crosstalk statistics match).
                 Slower (~2x "qr": numpy-side Box-Muller instead of torch's
                 native sampler) -- use when the tighter assumption matters
                 more than per-packet key-generation speed.
    mode="sign": random +-1 diagonal (D,) -- orthogonal and involutory
                 (bind == unbind), O(D) storage. Use when the slot count is
                 large enough that caching QR matrices is a memory concern.

    seed accepts either an int (PUBLIC multiplexing: key = f(seed, public
    slot), see module docstring) or a dict[slot, secret] (PRIVATE broadcast
    encryption: key = the slot's own secret, see mint_receiver_secrets).
    """

    def __init__(self, dim: int, seed: int | dict[int, int] = 1234, mode: str = "qr",
                 cache_limit: int = 64, nonce: int | None = None,
                 row_keys: bool = False):
        if mode not in ("qr", "qr_shake", "sign"):
            raise ValueError(f"Unknown keyring mode: {mode!r}")
        self.dim = dim
        self.seed = seed
        self.mode = mode
        self.cache_limit = cache_limit
        self.nonce = nonce
        self.row_keys = row_keys
        self._cache: dict[tuple[int, int], Any] = {}

    def _slot_seed(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(
                    f"slot {slot} has no enrolled private secret in this "
                    f"keyring (private mode: only slots {sorted(self.seed)} "
                    f"are known here)"
                )
            secret = self.seed[slot]
            if self.nonce is not None:
                return _derive_nonced_seed(secret, self.nonce)
            return secret
        # Large odd multiplier decorrelates neighbouring (seed, slot) pairs.
        return (self.seed * 1_000_003 + slot * 7_919) % (2**31 - 1)

    def key(self, slot: int, row: int = 0):
        """Key for (slot, row). `row` is ignored unless row_keys=True, in
        which case each latent row gets an INDEPENDENT key -- required for
        the IND-CPA claim whenever K > 1, see _derive_row_seed."""
        import torch

        cache_key = (slot, row if self.row_keys else 0)
        if cache_key in self._cache:
            return self._cache[cache_key]
        base = self._slot_seed(slot)
        seed = _derive_row_seed(base, row) if self.row_keys else base
        if self.mode == "sign":
            gen = torch.Generator().manual_seed(seed)
            k = torch.where(
                torch.rand(self.dim, generator=gen) < 0.5,
                torch.tensor(-1.0), torch.tensor(1.0),
            )
        else:
            if self.mode == "qr_shake":
                g = _shake_gaussian_matrix(seed, self.dim, self.dim)
            else:
                gen = torch.Generator().manual_seed(seed)
                g = torch.randn(self.dim, self.dim, generator=gen)
            q, r = torch.linalg.qr(g)
            # Sign-fix so Q is the unique canonical factor (deterministic
            # across LAPACK builds up to float rounding).
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
            k = q
        if len(self._cache) >= self.cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = k
        return k

    def bind(self, latents, slot: int, n_slots: int | None = None):
        """latents: (K, D) float32 -> bound (K, D) float32.

        n_slots is accepted for interface parity with SubspaceKeyring and
        ignored: rotation keys are load-independent.

        With row_keys=True each row k is bound under its OWN key Q^(k), which
        is what stops the plaintext Gram matrix from passing through the
        cipher intact (_derive_row_seed). Superposition is unaffected: the
        packet sum is still row-wise, so unbind still recovers row k of the
        intended slot plus row-k crosstalk.
        """
        import torch

        z = latents.to(torch.float32)
        if not self.row_keys:
            return z * self.key(slot) if self.mode == "sign" else z @ self.key(slot)
        rows = []
        for k in range(z.shape[0]):
            kk = self.key(slot, k)
            rows.append(z[k] * kk if self.mode == "sign" else z[k] @ kk)
        return torch.stack(rows)

    def unbind(self, packet, slot: int, n_slots: int | None = None):
        """packet: (K, D) float32 -> unbound (K, D) float32."""
        import torch

        p = packet.to(torch.float32)
        if not self.row_keys:
            return p * self.key(slot) if self.mode == "sign" else p @ self.key(slot).T
        rows = []
        for k in range(p.shape[0]):
            kk = self.key(slot, k)
            rows.append(p[k] * kk if self.mode == "sign" else p[k] @ kk.T)
        return torch.stack(rows)


class SubspaceKeyring:
    """Disjoint orthogonal subspaces per slot: zero-crosstalk superposition.

    One master orthogonal basis Q (D x D, seed-deterministic) is split by
    columns; in a packet shared by N slots, slot j owns columns
    [j*D//N, (j+1)*D//N). bind() projects latents into the slot's subspace
    (within the ambient D-dim space), so the packet sum is a DIRECT sum and
    unbind() recovers each slot's projection exactly, at any load.

    Trade-off vs OrthogonalKeyring ("qr" mode): capacity is a hard split
    (D/N clean dims per slot) instead of a soft one (D dims polluted by
    N-1 messages of crosstalk). Reconstruction quality is then set by how
    well the codec survives a rank-D/N projection -- a plain bottleneck,
    which is an easier training target than crosstalk denoising.

    Unlike rotation keys, bind/unbind REQUIRE n_slots (the subspace width
    depends on how many slots share the packet), and slot ids must be
    0..n_slots-1.

    NOT confidentiality-grade: every slot's basis is a public deterministic
    slice of ONE shared master matrix, so anyone who knows the algorithm,
    the (public, small) master seed, and n_slots can compute every slot's
    key -- an eavesdropper just tries all N <= ~16 slot indices and reads
    off whichever decode is fluent. Good for capacity/crosstalk-free
    multiplexing where confidentiality was never the point (see module
    docstring's PUBLIC mode); for "receiver decodes only its own message",
    use OrthogonalKeyring/RandomSubspaceKeyring/FeistelKeyring with a
    dict-seed (private per-receiver secret, unenumerable) instead.
    """

    def __init__(self, dim: int, seed: int = 1234):
        self.dim = dim
        self.seed = seed
        self._master = None

    def master(self):
        import torch

        if self._master is None:
            gen = torch.Generator().manual_seed(self.seed)
            g = torch.randn(self.dim, self.dim, generator=gen)
            q, r = torch.linalg.qr(g)
            self._master = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        return self._master

    def basis(self, slot: int, n_slots: int):
        """(D, D//n_slots) orthonormal basis of slot's subspace."""
        if not 0 <= slot < n_slots:
            raise ValueError(f"slot {slot} outside 0..{n_slots - 1}")
        width = self.dim // n_slots
        return self.master()[:, slot * width:(slot + 1) * width]

    def bind(self, latents, slot: int, n_slots: int | None = None):
        import torch

        if n_slots is None:
            raise ValueError("SubspaceKeyring.bind requires n_slots")
        b = self.basis(slot, n_slots)
        return (latents.to(torch.float32) @ b) @ b.T

    def unbind(self, packet, slot: int, n_slots: int | None = None):
        # Projection is idempotent and slot subspaces are disjoint, so
        # unbinding equals binding applied to the packet.
        return self.bind(packet, slot, n_slots)


class RandomSubspaceKeyring:
    """Independent private random subspace per slot: no shared master, no N.

    Unlike SubspaceKeyring (one shared master basis sliced into disjoint,
    globally-coordinated blocks), each slot's (D, width) basis is drawn
    from its OWN (seed, slot) pair -- nobody else's key is needed to build
    or use it, and slots do not partition a fixed ambient space, so a
    sender never needs to know N (how many others are sharing the packet).

    Two independent random `width`-dim subspaces of a `dim`-dim ambient
    space have expected pairwise overlap that concentrates around
    width^2 / dim (Grassmannian / random-subspace packing), so crosstalk
    shrinks smoothly as width shrinks and grows smoothly with more slots,
    instead of SubspaceKeyring's exact-zero/hard-wall behavior. A sender
    can unilaterally shrink `width` to cut its own footprint on everyone
    else, trading it against its own message's capacity -- no negotiation
    with other senders required.

    bind/unbind are the same idempotent orthogonal projection as
    SubspaceKeyring (bind == unbind); n_slots is accepted for interface
    parity and ignored -- width is fixed at construction, not derived
    from load.

    seed accepts either an int (PUBLIC, key = f(seed, public slot)) or a
    dict[slot, secret] (PRIVATE, see OrthogonalKeyring's docstring and
    mint_receiver_secrets) -- this is the natural private-mode partner to
    OrthogonalKeyring's rotation: unlike SubspaceKeyring, each slot's basis
    is already independently drawn with no shared master, so plugging in
    per-receiver secrets costs nothing extra.
    """

    def __init__(self, dim: int, width: int, seed: int | dict[int, int] = 1234,
                 cache_limit: int = 64, nonce: int | None = None):
        if not 0 < width <= dim:
            raise ValueError(f"width must be in (0, {dim}], got {width}")
        self.dim = dim
        self.width = width
        self.seed = seed
        self.cache_limit = cache_limit
        self.nonce = nonce
        self._cache: dict[int, Any] = {}

    def _slot_seed(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(
                    f"slot {slot} has no enrolled private secret in this "
                    f"keyring (private mode: only slots {sorted(self.seed)} "
                    f"are known here)"
                )
            secret = self.seed[slot]
            if self.nonce is not None:
                return _derive_nonced_seed(secret, self.nonce)
            return secret
        return (self.seed * 1_000_003 + slot * 7_919) % (2**31 - 1)

    def basis(self, slot: int, n_slots: int | None = None):
        """(dim, width) orthonormal basis, private to this slot."""
        import torch

        if slot in self._cache:
            return self._cache[slot]
        gen = torch.Generator().manual_seed(self._slot_seed(slot))
        g = torch.randn(self.dim, self.width, generator=gen)
        q, r = torch.linalg.qr(g)
        b = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        if len(self._cache) >= self.cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[slot] = b
        return b

    def bind(self, latents, slot: int, n_slots: int | None = None):
        import torch

        b = self.basis(slot)
        return (latents.to(torch.float32) @ b) @ b.T

    def unbind(self, packet, slot: int, n_slots: int | None = None):
        # Same reasoning as SubspaceKeyring: projection is idempotent, so
        # binding and unbinding are the same operator. Slots are NOT
        # guaranteed disjoint here, so (unlike SubspaceKeyring) this is an
        # approximate recovery, not exact.
        return self.bind(packet, slot, n_slots)


class FeistelKeyring:
    """Keyed NONLINEAR invertible bind/unbind -- a small Feistel network.

    OrthogonalKeyring/SubspaceKeyring are keyed LINEAR maps (z @ Q). Linear
    ciphers are broken by known/chosen-plaintext linear algebra (solve for
    Q from enough (z, bind(z)) pairs) and by key reuse (repeated Q is a
    multi-time pad). This class replaces the per-slot transform with a
    keyed nonlinear bijection so recovering it from plaintext/ciphertext
    pairs is not a linear-algebra exercise.

    Design mirrors a standard cipher SP-network, split into a PUBLIC part
    (round-function architecture + weights, and the inter-round
    permutation -- known to everyone, like a published cipher spec) and a
    PRIVATE part (the per-slot key vector fed as conditioning input, like
    a cipher key):

      for each of `n_rounds` rounds:
        x_active = x_active + f_r(x_other, key_vec)   # additive coupling:
                                                        # invertible for ANY
                                                        # f_r, keyed or not
        x = permute(x, round_perm)                     # PUBLIC fixed
                                                        # shuffle; pure
                                                        # diffusion, carries
                                                        # no secret so it
                                                        # gives an attacker
                                                        # nothing to solve

    f_r is a small fixed-random-weight 2-layer MLP (tanh-bounded, weights
    are PUBLIC and shared across all slots -- only `key_vec`, regenerated
    per-slot from the private seed exactly like OrthogonalKeyring._slot_seed,
    differs between receivers). Nothing is trained here: this class is a
    drop-in swap for the *linear* keyed maps, still evaluated zero-shot
    against codecs trained under linear crosstalk.

    IMPORTANT CAVEAT (why this is not simply "better"): superposition sums
    BOUND latents across parties, Z = sum_i bind_i(z_i). For linear bind,
    unbind_j(Z) = Q_j^T Z = z_j + sum_{i!=j} Q_j^T Q_i z_i distributes
    cleanly into signal + linear crosstalk. bind_j here is nonlinear, so
    unbind_j(Z) = bind_j^{-1}(sum_i bind_i(z_i)) does NOT distribute --
    there is no algebraic guarantee unbind_j(Z) is close to z_j at all once
    other parties' bound latents are comparable in magnitude to bind_j(z_j)
    (only a local/small-perturbation regime behaves like signal+noise, via
    a first-order Jacobian argument). Single-slot round-trip (N=1, no
    summation) is exactly lossless by construction; multi-party recovery
    quality is an open empirical question this class exists to measure,
    not an assumed property.
    """

    def __init__(self, dim: int, seed: int | dict[int, int] = 1234, n_rounds: int = 4,
                 key_dim: int = 64, hidden_dim: int = 128,
                 arch_seed: int = 20260721, weights_path: str | None = None,
                 nonce: int | None = None):
        self.dim = dim
        self.seed = seed
        self.n_rounds = n_rounds
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.arch_seed = arch_seed
        self.nonce = nonce
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self._round_fns = None
        self._mixes = None
        self._key_cache: dict[int, Any] = {}
        if weights_path is not None:
            self.load_weights(weights_path)

    def save_weights(self, path: str):
        """Persist round-function weights (+ mixing matrices, unchanged
        from the random-init ones) so a version TRAINED for key-
        decorrelation (training.programs.train_feistel_keyring) can be reloaded
        instead of the fixed-random-init defaults."""
        import torch

        self._build_public_arch()
        torch.save({
            "dim": self.dim, "n_rounds": self.n_rounds,
            "key_dim": self.key_dim, "hidden_dim": self.hidden_dim,
            "round_fns": [tuple(t.detach().clone() for t in fn) for fn in self._round_fns],
            "mixes": [m.detach().clone() for m in self._mixes],
        }, path)

    def load_weights(self, path: str):
        import torch

        ckpt = torch.load(path, weights_only=True)
        for attr in ("dim", "n_rounds", "key_dim", "hidden_dim"):
            if ckpt[attr] != getattr(self, attr):
                raise ValueError(
                    f"Weight file {attr}={ckpt[attr]} does not match "
                    f"keyring {attr}={getattr(self, attr)}")
        self._round_fns = ckpt["round_fns"]
        self._mixes = ckpt["mixes"]

    def _build_public_arch(self):
        """Fixed-weight round MLPs + fixed dense mixing matrices. PUBLIC:
        same for every slot/key, deterministic from arch_seed only (never
        the private per-slot seed).

        Two failed iterations before this one, kept here because the
        failure mode is the whole point of building this class:
        1. Index PERMUTATION between rounds only reorders values, never
           blends them -- wrong-key recovery stayed >0.4 cosine-similar
           to the true latents even at 32 rounds. Replaced with a fixed
           PUBLIC random orthogonal mixing matrix per round (the
           "MixColumns" role in a real SP-network -- linear but essential
           for diffusion, public because it carries no secret).
        2. Feeding `key` by CONCATENATING it into the round MLP's input
           barely helped (wrong-key cos-sim only dropped by feeding key
           through a giant D-dim input where it's diluted/saturated by
           x -- isolated test: cos-sim(f(x,key0), f(x,key1)) = 0.45
           holding x fixed). Switched to FiLM-style conditioning (key
           produces a per-hidden-unit scale+shift applied to x's own
           hidden representation, not concatenated with it) -- same
           isolated test: cos-sim drops to 0.06. Key needs to modulate
           the transform, not compete with x for a shared input slot."""
        import torch

        if self._round_fns is not None:
            return
        gen = torch.Generator().manual_seed(self.arch_seed)
        fns = []
        mixes = []
        for r in range(self.n_rounds):
            in_dim = self.d2 if r % 2 == 0 else self.d1
            out_dim = self.d1 if r % 2 == 0 else self.d2
            w1 = torch.randn(in_dim, self.hidden_dim, generator=gen) / in_dim ** 0.5
            w2 = torch.randn(self.hidden_dim, out_dim, generator=gen) / self.hidden_dim ** 0.5
            g_w = torch.randn(self.key_dim, self.hidden_dim, generator=gen) / self.key_dim ** 0.5
            b_w = torch.randn(self.key_dim, self.hidden_dim, generator=gen) / self.key_dim ** 0.5
            fns.append((w1, w2, g_w, b_w))
            g = torch.randn(self.dim, self.dim, generator=gen)
            q, rr = torch.linalg.qr(g)
            q = q * torch.sign(torch.diagonal(rr)).unsqueeze(0)
            mixes.append(q)
        self._round_fns = fns
        self._mixes = mixes

    def _round_fn(self, r: int, x, key_vec):
        """f_r(x, key) -> bounded additive update. FiLM conditioning: key
        drives a per-hidden-unit (scale, shift) applied to x's own hidden
        projection, so the key modulates the transform rather than
        competing with x inside a shared input vector (see
        _build_public_arch's docstring for why concatenation failed)."""
        import torch

        w1, w2, g_w, b_w = self._round_fns[r]
        h = x @ w1
        # gamma must be zero-mean across random keys: an earlier version
        # used tanh(.)*2+1 (mean ~1), which gave every key a shared
        # "baseline" scaling and left a persistent ~0.13 mean cosine
        # correlation between DIFFERENT keys' outputs (measured directly)
        # -- a systematic leak, not sampling noise. tanh(.)*2 is
        # zero-mean, and the measured mean drops to ~0.
        gamma = torch.tanh(key_vec @ g_w) * 2
        beta = torch.tanh(key_vec @ b_w) * 2
        h = torch.tanh(h * gamma + beta)
        return h @ w2

    def _slot_seed(self, slot: int) -> int:
        if isinstance(self.seed, dict):
            if slot not in self.seed:
                raise KeyError(
                    f"slot {slot} has no enrolled private secret in this "
                    f"keyring (private mode: only slots {sorted(self.seed)} "
                    f"are known here)"
                )
            secret = self.seed[slot]
            if self.nonce is not None:
                return _derive_nonced_seed(secret, self.nonce)
            return secret
        return (self.seed * 1_000_003 + slot * 7_919) % (2**31 - 1)

    def key(self, slot: int):
        import torch

        if slot in self._key_cache:
            return self._key_cache[slot]
        gen = torch.Generator().manual_seed(self._slot_seed(slot))
        k = torch.randn(self.key_dim, generator=gen)
        self._key_cache[slot] = k
        return k

    def bind(self, latents, slot: int, n_slots: int | None = None):
        """latents: (K, D) float32 -> bound (K, D) float32."""
        import torch

        self._build_public_arch()
        z = latents.to(torch.float32)
        key_vec = self.key(slot).unsqueeze(0).expand(z.shape[0], -1)
        x1, x2 = z[:, :self.d1], z[:, self.d1:]
        for r in range(self.n_rounds):
            if r % 2 == 0:
                x1 = x1 + self._round_fn(r, x2, key_vec)
            else:
                x2 = x2 + self._round_fn(r, x1, key_vec)
            x = torch.cat([x1, x2], dim=-1) @ self._mixes[r]
            x1, x2 = x[:, :self.d1], x[:, self.d1:]
        return torch.cat([x1, x2], dim=-1)

    def unbind(self, packet, slot: int, n_slots: int | None = None):
        """packet: (K, D) float32 -> unbound (K, D) float32. Exact inverse
        of bind for a single-slot round-trip (bind then immediately
        unbind with the same key); see class docstring for why this is
        NOT guaranteed to isolate slot `slot`'s signal out of a real
        multi-party sum."""
        import torch

        self._build_public_arch()
        p = packet.to(torch.float32)
        key_vec = self.key(slot).unsqueeze(0).expand(p.shape[0], -1)
        x = p
        for r in reversed(range(self.n_rounds)):
            x = x @ self._mixes[r].T
            x1, x2 = x[:, :self.d1], x[:, self.d1:]
            if r % 2 == 0:
                x1 = x1 - self._round_fn(r, x2, key_vec)
            else:
                x2 = x2 - self._round_fn(r, x1, key_vec)
            x = torch.cat([x1, x2], dim=-1)
        return x


def build_keyring(dim: int, seed: int | dict[int, int] = 1234, mode: str = "qr",
                   width: int | None = None, **extra_kwargs):
    # Only OrthogonalKeyring implements per-row keys so far. The other
    # keyrings would need the same treatment before they could carry an
    # IND-CPA claim at K > 1 (see _derive_row_seed); rather than silently
    # accepting and ignoring the flag -- which would let a caller believe
    # they had asked for the fix and hand back a still-broken keyring --
    # reject it explicitly for those modes.
    row_keys = extra_kwargs.pop("row_keys", None)
    if row_keys and mode in ("subspace", "random_subspace", "feistel"):
        raise ValueError(
            f"row_keys=True is not implemented for mode={mode!r} (only 'qr'/"
            "'qr_shake'/'sign'). Without per-row keys, a K>1 message's Gram "
            "matrix passes through the cipher intact -- see _derive_row_seed. "
            "Use mode='qr' for the confidentiality-grade path."
        )
    if row_keys is not None and mode not in ("subspace", "random_subspace", "feistel"):
        extra_kwargs["row_keys"] = row_keys
    if mode == "subspace":
        if isinstance(seed, dict):
            raise ValueError(
                "SubspaceKeyring has no private mode: every slot's basis is "
                "sliced from ONE shared master matrix, so the split is "
                "public by construction (great for capacity-only "
                "multiplexing, not confidentiality -- see its class "
                "docstring). Use mode='qr', 'random_subspace', or "
                "'feistel' with a dict seed for private per-receiver keys."
            )
        return SubspaceKeyring(dim, seed=seed, **extra_kwargs)
    if mode == "random_subspace":
        if width is None:
            raise ValueError("random_subspace mode requires width")
        return RandomSubspaceKeyring(dim, width=width, seed=seed, **extra_kwargs)
    if mode == "feistel":
        return FeistelKeyring(dim, seed=seed, **extra_kwargs)
    return OrthogonalKeyring(dim, seed=seed, mode=mode, **extra_kwargs)


def superpose(keyring, latents_by_slot: dict[int, Any]):
    """Bind each (K, D) latent block with its slot key and sum into one packet."""
    n_slots = len(latents_by_slot)
    packet = None
    for slot, z in latents_by_slot.items():
        bound = keyring.bind(z, slot, n_slots)
        packet = bound if packet is None else packet + bound
    return packet


def serialize_packet(packet, n_slots: int = 1) -> str:
    """(K, D) float32 tensor -> 'K:D:N:<base64 fp32 bytes>'.

    N (the slot count) rides in the header so subspace unbinding always
    uses the width the packet was actually built with.
    """
    import torch

    arr = packet.detach().cpu().to(torch.float32).numpy()
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")
    return f"{arr.shape[0]}:{arr.shape[1]}:{n_slots}:{b64}"


def deserialize_packet(data: str):
    """Inverse of serialize_packet. Returns (CPU float32 tensor (K, D), n_slots)."""
    import torch

    parts = data.split(":", 3)
    if len(parts) != 4:
        raise ValueError("Malformed packet string")
    k, d, n_slots = int(parts[0]), int(parts[1]), int(parts[2])
    arr = np.frombuffer(base64.b64decode(parts[3]), dtype=np.float32).copy()
    return torch.from_numpy(arr.reshape(k, d)), n_slots


def latent_indices(seq_len: int, num_latents: int) -> list[int]:
    """Which encoder positions become latents (same rule as pretraining)."""
    if num_latents == 1:
        return [seq_len - 1]
    step = seq_len / num_latents
    return [min(int(i * step + step / 2), seq_len - 1) for i in range(num_latents)]


class LatentCodec:
    """Encode text -> (K, D) float32 latents and decode them back to text.

    Wraps the shared autoencoder LM checkpoint (same format as
    pretrain_autoencoder.py / pretrain_superpose.py save). If the checkpoint
    contains projection.pt, latents live in the bottleneck space (D =
    bottleneck_dim); otherwise D = hidden_size.

    The encode chat template, latent position rule, and RECONSTRUCT decode
    prompt intentionally mirror AutoencoderCompressor so a superpose-
    fine-tuned checkpoint stays drop-in compatible.
    """

    def __init__(
        self,
        model_path: str = "data/autoencoder_pretrain/final",
        num_latents: int | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        max_len: int = 384,
    ):
        self.model_path = model_path
        self.num_latents = num_latents
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_len = max_len
        self._model = None
        self._tokenizer = None
        self._proj_down = None
        self._proj_up = None

    @property
    def latent_dim(self) -> int:
        self._load()
        if self._proj_down is not None:
            return self._proj_down.out_features
        return self._model.config.hidden_size

    def _load(self):
        if self._model is not None:
            return self._model, self._tokenizer
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
        if self.num_latents is None:
            cfg_path = os.path.join(self.model_path, "ae_config.json")
            if os.path.exists(cfg_path):
                self.num_latents = json.load(open(cfg_path)).get("num_latents", 4)
            else:
                self.num_latents = 4
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        proj_path = os.path.join(self.model_path, "projection.pt")
        if os.path.exists(proj_path):
            ckpt = torch.load(proj_path, map_location=self.device, weights_only=True)
            H = self._model.config.hidden_size
            d = ckpt["bottleneck_dim"]
            self._proj_down = nn.Linear(H, d, dtype=torch.bfloat16).to(self.device)
            self._proj_up = nn.Linear(d, H, dtype=torch.bfloat16).to(self.device)
            self._proj_down.load_state_dict(ckpt["proj_down"])
            self._proj_up.load_state_dict(ckpt["proj_up"])
            self._proj_down.eval()
            self._proj_up.eval()
        return self._model, self._tokenizer

    def encode(self, text: str):
        """text -> (num_latents, D) float32 CPU tensor."""
        import torch

        model, tokenizer = self._load()
        enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        enc_ids = tokenizer.encode(
            enc_text, return_tensors="pt", truncation=True, max_length=self.max_len,
        ).to(self.device)
        seq_len = enc_ids.shape[1]
        last_layer = model.config.num_hidden_layers - 1

        with torch.no_grad():
            out = model(enc_ids, output_hidden_states=True)
            hidden = out.hidden_states[last_layer][0]
            idx = latent_indices(seq_len, self.num_latents)
            latents = hidden[idx]  # (K, H) bf16
            if self._proj_down is not None:
                latents = self._proj_down(latents)
        return latents.cpu().to(torch.float32)

    def decode(self, latents) -> str | None:
        """(num_latents, D) float32 latents -> reconstructed text."""
        import torch

        model, tokenizer = self._load()
        z = latents.to(self.device, dtype=torch.bfloat16)
        if self._proj_up is not None:
            z = self._proj_up(z)  # (K, H)

        num_latents = z.shape[0]
        dec_prompt = "<|im_start|>user\n"
        for i in range(num_latents):
            dec_prompt += f"<|L{i}|>"
        dec_prompt += "RECONSTRUCT<|im_end|>\n<|im_start|>assistant\n"

        prompt_ids = tokenizer.encode(dec_prompt, return_tensors="pt").to(self.device)
        embed_layer = model.get_input_embeddings()
        embeds = embed_layer(prompt_ids)
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        for i in range(num_latents):
            li_id = tokenizer.convert_tokens_to_ids(f"<|L{i}|>")
            positions = (prompt_ids[0] == li_id).nonzero(as_tuple=True)[0]
            for p in positions:
                embeds[0, p] = z[i]

        with torch.no_grad():
            past_kv = None
            current_emb = embeds
            generated: list[int] = []
            for _ in range(self.max_new_tokens):
                out = model(
                    inputs_embeds=current_emb, past_key_values=past_kv, use_cache=True,
                )
                past_kv = out.past_key_values
                next_id = out.logits[0, -1, :].argmax().item()
                if next_id == eos_id:
                    break
                generated.append(next_id)
                current_emb = embed_layer(
                    torch.tensor([[next_id]], device=self.device)
                )
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return decoded if decoded else None


class SuperposedPacketCodec:
    """Full pipeline: texts+slots -> one packet string; packet+slot -> text.

    PUBLIC/multiplexing mode (module docstring): key_seed is one shared
    value, so decode_slot(packet, j) is computable by anyone for any j --
    this class does not, by itself, stop a receiver from decoding a slot
    that is not theirs. For that guarantee, use SecureBroadcastCodec
    (sender) / SecureReceiverCodec (receiver) below instead, which are the
    same encode/superpose/decode pipeline keyed with private per-receiver
    secrets from mint_receiver_secrets.
    """

    def __init__(self, codec: LatentCodec, keyring: OrthogonalKeyring | None = None,
                 key_seed: int = 1234, key_mode: str = "qr"):
        self.codec = codec
        self._keyring = keyring
        self._key_seed = key_seed
        self._key_mode = key_mode

    @property
    def keyring(self):
        if self._keyring is None:
            self._keyring = build_keyring(
                self.codec.latent_dim, seed=self._key_seed, mode=self._key_mode,
            )
        return self._keyring

    def encode_packet(self, texts_by_slot: dict[int, str]) -> str:
        latents = {slot: self.codec.encode(t) for slot, t in texts_by_slot.items()}
        packet = superpose(self.keyring, latents)
        return serialize_packet(packet, n_slots=len(texts_by_slot))

    def decode_slot(self, packet_str: str, slot: int) -> str | None:
        packet, n_slots = deserialize_packet(packet_str)
        return self.codec.decode(self.keyring.unbind(packet, slot, n_slots))


def mint_receiver_secrets(n_slots: int, start_slot: int = 0) -> dict[int, int]:
    """Cryptographically random per-receiver secrets for the PRIVATE keying
    mode (module docstring; consumed by SecureBroadcastCodec /
    SecureReceiverCodec below).

    Drawn from Python's `secrets` module (os.urandom-backed CSPRNG), NOT a
    torch.Generator seeded from anything public -- the whole point is that
    slot j's key must not be computable from j, a shared seed, or any other
    slot's secret. The sender calls this once, keeps the full dict to build
    packets, and hands each receiver ONLY its own (slot, secret) pair
    out-of-band (however sender<->receiver key exchange happens outside
    this module -- e.g. a side channel, a KMS, a prior handshake).

    63 bits (fits torch.Generator.manual_seed's int64 range) is well beyond
    brute-force range (2**63), unlike the PUBLIC mode's effective keyspace
    of just N <= ~16 enumerable slot indices.
    """
    import secrets as _secrets

    return {start_slot + i: _secrets.randbits(63) for i in range(n_slots)}


def _with_nonce_prefix(nonce: int, packet_str: str) -> str:
    return f"{nonce}|{packet_str}"


def _split_nonce_prefix(s: str) -> tuple[int | None, str]:
    """Inverse of _with_nonce_prefix. '|' never appears in serialize_packet's
    output (base64 standard alphabet + ':' only), so splitting on the first
    '|' is unambiguous; a string with no '|' is a legacy (no-nonce) packet.
    """
    if "|" in s:
        head, rest = s.split("|", 1)
        try:
            return int(head), rest
        except ValueError:
            pass
    return None, s


class SecureBroadcastCodec:
    """Sender side of the private-keying pipeline: N different plaintext
    messages -> ONE packet, each bound under ITS OWN receiver's private
    secret (mint_receiver_secrets), summed together (superpose). Requires
    knowing every recipient's secret, same as any broadcast-encryption
    sender must enroll its recipients before it can address them.

    key_mode picks the underlying bind/unbind primitive: "qr" (linear
    rotation, cheap, matches most trained checkpoints) or "feistel" (keyed
    nonlinear coupling); pass weights_path via keyring_kwargs to use a
    key-decorrelation-trained round function, see
    training.programs.train_feistel_keyring. "subspace" is rejected by
    build_keyring -- see SubspaceKeyring's docstring for why it cannot be
    made private.

    nonce (module docstring's known-plaintext caveat) -- REQUIRED reading
    before setting this to anything but the default:
      True (default): mint a fresh CSPRNG nonce on EVERY encode_packet()
          call, so every packet uses a single-use derived key. This is the
          only setting that actually defends against the known-plaintext
          linear-solve attack for repeated use of the same secrets_by_slot.
      an int: use that FIXED nonce for every packet from this instance.
          Only for deterministic single-packet tests -- reusing one sender
          object with a fixed int nonce for MANY packets is exactly the
          same key-reuse vulnerability as nonce=False, just with an extra
          hash step that does not change the exposure.
      False / None: legacy no-nonce mode (secret used directly as the key,
          reused for every packet) -- kept only for backward compatibility
          with the (public) int-seed-style tests/scripts; NOT SAFE for
          repeated real use of the same secrets_by_slot.
    """

    def __init__(self, codec: LatentCodec, secrets_by_slot: dict[int, int],
                 key_mode: str = "qr", nonce: bool | int | None = True,
                 row_keys: bool = True, **keyring_kwargs):
        self.codec = codec
        self.secrets_by_slot = dict(secrets_by_slot)
        self._key_mode = key_mode
        # row_keys defaults to True on the SECURE path (unlike the raw
        # keyring, whose default stays False for bit-compatibility with the
        # public multiplexing use case that never claimed confidentiality):
        # with K > 1 latents per message, a shared Q leaks the plaintext's
        # whole Gram matrix and is a total IND-CPA break. See
        # _derive_row_seed and reports/crypto_provable_security_20260722.md
        # Sec 3a. Set False only to reproduce pre-fix (broken) behavior.
        # Only forwarded for modes that implement it; for others the flag is
        # meaningless rather than silently ignored (build_keyring rejects an
        # explicit row_keys=True there, see ROW_KEY_MODES).
        if key_mode in ROW_KEY_MODES:
            keyring_kwargs = dict(keyring_kwargs, row_keys=row_keys)
        self._keyring_kwargs = keyring_kwargs
        self._nonce_mode = nonce
        self._legacy_keyring = None  # cached only for nonce_mode in (False, None)

    def _resolve_nonce(self) -> int | None:
        if self._nonce_mode is True:
            import secrets as _secrets
            return _secrets.randbits(63)
        if self._nonce_mode is False or self._nonce_mode is None:
            return None
        return int(self._nonce_mode)

    def _keyring_for(self, nonce: int | None):
        if nonce is None:
            if self._legacy_keyring is None:
                self._legacy_keyring = build_keyring(
                    self.codec.latent_dim, seed=self.secrets_by_slot,
                    mode=self._key_mode, **self._keyring_kwargs,
                )
            return self._legacy_keyring
        return build_keyring(
            self.codec.latent_dim, seed=self.secrets_by_slot,
            mode=self._key_mode, nonce=nonce, **self._keyring_kwargs,
        )

    def encode_packet(self, texts_by_slot: dict[int, str]) -> str:
        if not set(texts_by_slot) <= set(self.secrets_by_slot):
            missing = set(texts_by_slot) - set(self.secrets_by_slot)
            raise KeyError(f"no enrolled secret for slot(s) {sorted(missing)}")
        nonce = self._resolve_nonce()
        keyring = self._keyring_for(nonce)
        latents = {slot: self.codec.encode(t) for slot, t in texts_by_slot.items()}
        packet = superpose(keyring, latents)
        packet_str = serialize_packet(packet, n_slots=len(texts_by_slot))
        if nonce is not None:
            return _with_nonce_prefix(nonce, packet_str)
        return packet_str


class SecureReceiverCodec:
    """Receiver side of the private-keying pipeline: holds ONLY its own
    (my_slot, my_secret) pair -- never the sender's full secrets_by_slot --
    so decoding a different slot is not merely against the intended usage,
    it structurally KeyErrors (the underlying keyring's secret dict has
    exactly one entry; see OrthogonalKeyring._slot_seed etc.).

    key_mode and any keyring_kwargs (e.g. n_rounds, weights_path for
    "feistel") must match what SecureBroadcastCodec used to build the
    packet. The per-packet nonce (if any) is read from the packet itself
    (public, like an IV -- see SecureBroadcastCodec's docstring) and used
    to re-derive that packet's single-use key; nothing needs to be passed
    here beyond the receiver's own long-term secret.
    """

    def __init__(self, codec: LatentCodec, my_slot: int, my_secret: int,
                 key_mode: str = "qr", row_keys: bool = True, **keyring_kwargs):
        self.codec = codec
        self.my_slot = my_slot
        self.my_secret = my_secret
        self._key_mode = key_mode
        # Must match the sender's setting; both default True (see
        # SecureBroadcastCodec.__init__ for why, incl. the mode gating).
        if key_mode in ROW_KEY_MODES:
            keyring_kwargs = dict(keyring_kwargs, row_keys=row_keys)
        self._keyring_kwargs = keyring_kwargs

    def _keyring_for(self, nonce: int | None):
        kwargs = dict(self._keyring_kwargs)
        if nonce is not None:
            kwargs["nonce"] = nonce
        return build_keyring(
            self.codec.latent_dim, seed={self.my_slot: self.my_secret},
            mode=self._key_mode, **kwargs,
        )

    def decode(self, packet_str: str) -> str | None:
        nonce, inner = _split_nonce_prefix(packet_str)
        packet, n_slots = deserialize_packet(inner)
        keyring = self._keyring_for(nonce)
        unbound = keyring.unbind(packet, self.my_slot, n_slots)
        return self.codec.decode(unbound)
