"""
Token-level filtering policy for representational-match compression.

A lightweight MLP head sits on top of a frozen Qwen3-4B proxy model. For each
token in the input, the head outputs an importance score from the token's hidden
state at a target layer. During training, the top-k tokens by score are kept
(exact, deterministic) and the reward is purely representational similarity
(no compression penalty — the fixed rate handles that). During inference,
a threshold on the score selects tokens.

Gradient flows through a sigmoid-threshold pseudo-log-prob: treating the top-k
cut as a Bernoulli at the boundary score.

Key difference from the Bernoulli-sampling approach: the policy cannot collapse
to "delete everything" because it MUST keep exactly k tokens. It only chooses
which tokens are most important.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = AutoTokenizer = None


class PolicyHead(nn.Module):
    """MLP that maps token hidden states to scalar importance scores (logits).

    Higher score = token is more important to keep. No sigmoid — the scores
    are used directly for top-k selection.
    """

    def __init__(self, hidden_dim: int = 2560, ff_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


def build_compressed_text(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    keep_mask: torch.Tensor,
) -> tuple[str | None, float]:
    """Decode the kept token IDs to a text string.

    Returns (text, kept_ratio). Returns (None, ratio) if no tokens are kept
    or only BOS/EOS survive.
    """
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    mask = keep_mask.tolist() if isinstance(keep_mask, torch.Tensor) else list(keep_mask)

    kept_ids = []
    for tid, keep in zip(ids, mask):
        if keep < 0.5:
            continue
        if tid == bos_id or tid == eos_id:
            continue
        kept_ids.append(tid)

    kept_ratio = len(kept_ids) / max(len(ids), 1)
    if not kept_ids:
        return None, kept_ratio

    text = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
    return text if text else None, kept_ratio


def compute_representation(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    device: torch.device,
    max_length: int = 4096,
) -> torch.Tensor:
    """Normalized last-token hidden state at the target layer."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer][0, -1, :].float()
    return F.normalize(hs, dim=0)


def compute_sim_reward(h_comp: torch.Tensor, h_orig: torch.Tensor) -> torch.Tensor:
    """Non-negative similarity: max(0, -log(L2^2 + eps)), capped at 5.0."""
    l2_sq = ((h_comp - h_orig) ** 2).sum()
    raw = -torch.log(l2_sq + 1e-4)
    return torch.clamp(raw, min=0.0, max=5.0)


def _topk_mask_and_logprobs(
    scores: torch.Tensor, k: int, noise_scale: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k hard mask + pseudo-log-probs for gradient flow.

    With noise_scale > 0, adds Gumbel noise to scores before top-k selection
    for exploration. Multiple calls with the same scores + different noise
    produce different top-k masks — enabling group-normalized REINFORCE.

    Pseudo-log-probs treat each token as an independent Bernoulli whose
    threshold is the k-th score (after noise). Tokens above threshold get
    log(σ(score - τ)), below get log(σ(τ - score)).

    Args:
        scores: (seq_len,) raw scores from policy head.
        k: number of tokens to keep.
        noise_scale: std of Gumbel noise. 0 = deterministic top-k.

    Returns:
        (keep_mask, log_probs) each (seq_len,).
    """
    seq_len = scores.shape[0]
    k = min(k, seq_len)

    if noise_scale > 0:
        gumbel = -torch.log(-torch.log(torch.rand_like(scores).clamp(min=1e-8)) + 1e-8)
        noisy = scores + noise_scale * gumbel
    else:
        noisy = scores

    # Hard mask: top-k tokens by (possibly noisy) scores
    _, topk_indices = torch.topk(noisy, k)
    keep_mask = torch.zeros(seq_len, device=scores.device)
    keep_mask[topk_indices] = 1.0

    # Pseudo-log-probs: threshold at the k-th score from CLEAN scores
    # (so gradient doesn't flow through the noise)
    sorted_clean, _ = torch.sort(scores, descending=True)
    threshold = sorted_clean[k - 1] if k > 0 else sorted_clean[0] + 1.0

    diff = scores - threshold.detach()
    log_prob_keep = F.logsigmoid(diff)
    log_prob_drop = F.logsigmoid(-diff)

    log_probs = keep_mask * log_prob_keep + (1 - keep_mask) * log_prob_drop
    return keep_mask, log_probs


class TokenFilterModel:
    """Wraps a frozen Qwen3-4B model + trainable policy head for token filtering.

    Manages:
      - Loading/offloading the proxy model
      - Forward passes for per-token hidden states
      - Top-k keep/drop selection
      - Computing compressed representations and rewards
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        layer: int = 14,
        device: str | None = None,
        max_length: int = 4096,
    ):
        self.model_name = model_name
        self.layer = layer
        self.max_length = max_length

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None
        self.head: PolicyHead | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("TokenFilter requires the full Conda environment")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        hidden_dim = self.model.config.hidden_size
        self.head = PolicyHead(hidden_dim=hidden_dim).to(self.device)

    def unload_model(self) -> None:
        if self.model is not None:
            self.model.cpu()
            self.model = None
            torch.cuda.empty_cache()

    def get_token_hidden_states(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, per-token hidden states at target layer)."""
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
            h = out.hidden_states[self.layer]
        return enc["input_ids"][0], h[0]

    def get_representation(self, text: str) -> torch.Tensor:
        return compute_representation(
            text, self.model, self.tokenizer, self.layer, self.device, self.max_length
        )

    def rollout(
        self,
        text: str,
        keep_rate: float = 0.4,
        num_samples: int = 1,
        noise_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Deterministic rollout(s) at a fixed keep rate, with optional Gumbel exploration.

        Args:
            text: the view text.
            keep_rate: fraction of tokens to keep (0.0-1.0).
            num_samples: number of rollouts (for group-normalized REINFORCE).
            noise_scale: Gumbel noise std for exploration. Set higher for more
                         diverse masks across num_samples.

        Returns:
            dict with "keep_masks", "log_probs", "rewards", "kept_ratios", "scores".
            Each tensor has batch dim = num_samples.
        """
        input_ids, h_tokens = self.get_token_hidden_states(text)
        h_orig = compute_representation(
            text, self.model, self.tokenizer, self.layer, self.device, self.max_length
        )

        scores = self.head(h_tokens.float())
        seq_len = scores.shape[0]
        k = max(1, int(seq_len * keep_rate))

        all_masks = []
        all_log_probs = []
        all_rewards = []
        all_kept_ratios = []

        for _ in range(num_samples):
            keep_mask, log_probs = _topk_mask_and_logprobs(scores, k, noise_scale)
            comp_text, kept_ratio = build_compressed_text(input_ids, self.tokenizer, keep_mask)
            kept_ratio_t = torch.tensor(kept_ratio, device=self.device)

            if comp_text and kept_ratio > 0:
                h_comp = self.get_representation(comp_text)
                sim_reward = compute_sim_reward(h_comp, h_orig)
            else:
                sim_reward = torch.tensor(0.0, device=self.device)

            all_masks.append(keep_mask)
            all_log_probs.append(log_probs)
            all_rewards.append(sim_reward)
            all_kept_ratios.append(kept_ratio_t)

        return {
            "keep_masks": torch.stack(all_masks),
            "log_probs": torch.stack(all_log_probs),
            "rewards": torch.stack(all_rewards),
            "kept_ratios": torch.stack(all_kept_ratios),
            "scores": scores,
        }
