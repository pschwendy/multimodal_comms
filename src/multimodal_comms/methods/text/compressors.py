"""
Communication channel between agents.

The channel controls two orthogonal aspects of inter-agent communication:

1. Protocol: which raw discussion messages a receiver is shown at its turn.
   - "full_history": every turn re-transmits the entire discussion so far
     (the original HiddenBench protocol).
   - "delta": each agent only receives messages it has not yet seen
     (a lossless, less-redundant variant).

2. Compressor: a transformation applied to the transmitted view
   (identity, sliding window, embedding-based novelty filter,
   LLMLingua-2 token pruning).

The default configuration (full_history + identity) reproduces the original
HiddenBench protocol exactly.

Compressors are middle-layer modules: they may use their own open models but
never touch the (potentially proprietary) sender/receiver models, and they
never see task labels or hidden-information annotations.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from multimodal_comms.methods.predictive.certificate import certificate_scores, passes_certificate
from multimodal_comms.methods.predictive.model_diff import DiffOp, PDiffCodec


@dataclass
class ChannelStats:
    """Byte/message accounting for channel traffic."""
    # Content produced by senders (each message counted once)
    raw_messages: int = 0
    raw_chars: int = 0
    # Content actually placed into receivers' prompts (after protocol + compression)
    transmitted_messages: int = 0
    transmitted_chars: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "raw_messages": self.raw_messages,
            "raw_chars": self.raw_chars,
            "transmitted_messages": self.transmitted_messages,
            "transmitted_chars": self.transmitted_chars,
        }

    @staticmethod
    def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {k: after[k] - before[k] for k in after}


class Compressor(ABC):
    """Transforms a view (list of message dicts) before transmission.

    For compress-only compressors the default decompress() is a no-op.
    Compressors that re-encode content (like an autoencoder) should
    override both compress() and decompress().
    """

    name = "base"

    def set_task_context(self, options: list[str]) -> None:
        """Give the compressor public task context (the answer options).

        Options are public knowledge shared by all agents, so using them to
        protect key tokens from deletion is not information leakage.
        """
        pass

    def get_system_prompt_suffix(self) -> str:
        """Return text to append to the system prompt (sent once per agent).

        Grammar-based compressors can return a codebook here so the receiver
        LLM learns the compact symbols once, rather than per message.
        """
        return ""

    @abstractmethod
    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        """Return the (possibly reduced) list of message dicts to transmit."""
        ...

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        """Reverse the compression. Default: no-op (messages are already readable)."""
        return messages


class IdentityCompressor(Compressor):
    """No-op compressor (original behavior)."""

    name = "identity"

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        return messages


class WindowCompressor(Compressor):
    """Keep only messages from the last N rounds present in the view."""

    name = "window"

    def __init__(self, window_rounds: int = 2):
        self.window_rounds = window_rounds

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        max_round = max(m["round_num"] for m in messages)
        cutoff = max_round - self.window_rounds + 1
        return [m for m in messages if m["round_num"] >= cutoff]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class NoveltyCompressor(Compressor):
    """Drop sentences semantically redundant with earlier content in the view.

    Uses a small open sentence-embedding model (on CPU, so it never competes
    with the serving GPU). Sentences are kept in chronological order; a
    sentence is dropped when its cosine similarity to any already-kept
    sentence exceeds the threshold.

    With stateful=True the filter additionally remembers, per receiver, every
    sentence that receiver has already been shown (across turns) and drops
    repeats and paraphrases of them. This is a pure middleware analogue of the
    delta protocol at sub-message granularity: the protocol itself is left
    untouched, only the transmitted content shrinks. State is reset per task
    via set_task_context.
    """

    name = "novelty"

    def __init__(self, threshold: float = 0.85, stateful: bool = False):
        self.threshold = threshold
        self.stateful = stateful
        self._model = None
        # receiver_id -> list of embeddings already shown to that receiver
        self._receiver_memory: dict[int, list[Any]] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._receiver_memory = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
        return self._model

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import numpy as np

        model = self._get_model()

        # Split all messages into sentences, remembering origin
        per_message_sentences: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message_sentences.append(sentences)
            all_sentences.extend(sentences)

        if not all_sentences:
            return messages

        embeddings = model.encode(all_sentences, normalize_embeddings=True)

        # Seed with what this receiver has already been shown (stateful mode)
        kept_embeddings: list[Any] = []
        if self.stateful:
            kept_embeddings = list(self._receiver_memory.get(receiver_id, []))

        keep_flags: list[bool] = []
        for emb in embeddings:
            if kept_embeddings:
                sims = np.dot(np.stack(kept_embeddings), emb)
                if float(sims.max()) >= self.threshold:
                    keep_flags.append(False)
                    continue
            kept_embeddings.append(emb)
            keep_flags.append(True)

        if self.stateful:
            self._receiver_memory[receiver_id] = kept_embeddings

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message_sentences):
            kept = [
                s for s, flag in zip(sentences, keep_flags[cursor:cursor + len(sentences)]) if flag
            ]
            cursor += len(sentences)
            if kept:
                new_m = dict(m)
                new_m["content"] = " ".join(kept)
                result.append(new_m)
        return result


class LLMLingua2Compressor(Compressor):
    """Token-level pruning of message contents via LLMLingua-2.

    Content-neutral extractive compression using a small open encoder.
    Answer-option words are protected from deletion (they are public
    knowledge, and dropping them would corrupt votes as an artifact).
    """

    name = "llmlingua2"

    def __init__(self, rate: float = 0.4, min_chars: int = 80, device: str | None = None):
        self.rate = rate
        self.min_chars = min_chars
        self.device = device
        self._compressor = None
        self._force_tokens: list[str] = ["\n", ".", "!", "?", ","]

    def set_task_context(self, options: list[str]) -> None:
        words = set()
        for option in options:
            words.update(option.split())
        self._force_tokens = ["\n", ".", "!", "?", ","] + sorted(words)

    def _get_compressor(self):
        if self._compressor is None:
            import torch
            from llmlingua import PromptCompressor

            if self.device is None:
                # Prefer a GPU that isn't serving the model (GPU 0)
                if torch.cuda.device_count() > 1:
                    self.device = "cuda:1"
                elif torch.cuda.is_available():
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"

            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map=self.device,
            )
        return self._compressor

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        compressor = self._get_compressor()
        result = []
        for m in messages:
            content = m["content"]
            if len(content) < self.min_chars:
                result.append(m)
                continue
            out = compressor.compress_prompt(
                content,
                rate=self.rate,
                force_tokens=self._force_tokens,
            )
            new_m = dict(m)
            new_m["content"] = out["compressed_prompt"]
            result.append(new_m)
        return result


class LearnedSelectCompressor(Compressor):
    """Learned extractive sentence selection under a character budget.

    A trained classifier (open sentence encoder + logistic head, trained on
    transcripts of held-out tasks) scores each sentence's decision-relevance;
    the top-scoring sentences are kept, in chronological order, until the
    budget (rate x total chars) is reached. Purely extractive: the middleware
    can only delete sender content, never write, so it cannot inject
    reasoning or answers.

    Optionally composes with the stateful per-receiver dedup memory
    (drop already-shown content first, then select under budget).
    """

    name = "learned"

    def __init__(
        self,
        rate: float = 0.5,
        model_path: str = "data/selector_model.joblib",
        dedup: bool = False,
        dedup_threshold: float = 0.85,
    ):
        self.rate = rate
        self.model_path = model_path
        self.dedup = dedup
        self.dedup_threshold = dedup_threshold
        self._encoder = None
        self._classifier = None
        self._receiver_memory: dict[int, list[Any]] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._receiver_memory = {}

    def _load(self):
        if self._classifier is None:
            import joblib
            from sentence_transformers import SentenceTransformer
            bundle = joblib.load(self.model_path)
            self._encoder = SentenceTransformer(bundle["encoder_name"], device="cpu")
            self._classifier = bundle["classifier"]
        return self._encoder, self._classifier

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import numpy as np

        encoder, classifier = self._load()

        per_message: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message.append(sentences)
            all_sentences.extend(sentences)
        if not all_sentences:
            return messages

        embeddings = encoder.encode(all_sentences, normalize_embeddings=True)
        scores = classifier.predict_proba(embeddings)[:, 1]

        # Optional stateful dedup: zero out sentences this receiver has seen
        alive = np.ones(len(all_sentences), dtype=bool)
        if self.dedup:
            memory = list(self._receiver_memory.get(receiver_id, []))
            for i, emb in enumerate(embeddings):
                if memory:
                    sims = np.dot(np.stack(memory), emb)
                    if float(sims.max()) >= self.dedup_threshold:
                        alive[i] = False
                        continue
                memory.append(emb)
            self._receiver_memory[receiver_id] = memory

        # Budget: keep highest-scoring alive sentences within rate of the
        # ORIGINAL total chars (so dedup savings count toward the budget)
        total_chars = sum(len(s) for s in all_sentences)
        budget = self.rate * total_chars
        keep = np.zeros(len(all_sentences), dtype=bool)
        used = 0.0
        for i in np.argsort(-scores):
            if not alive[i]:
                continue
            cost = len(all_sentences[i])
            if used + cost > budget and used > 0:
                continue
            keep[i] = True
            used += cost

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message):
            kept = [s for s, flag in
                    zip(sentences, keep[cursor:cursor + len(sentences)]) if flag]
            cursor += len(sentences)
            if kept:
                new_m = dict(m)
                new_m["content"] = " ".join(kept)
                result.append(new_m)
        return result


class CounterfactualImportanceCompressor(Compressor):
    """Extractive compression via a distilled counterfactual-importance scorer.

    Importance of a sentence is defined as the KL divergence the receiver's
    belief distribution over answer options undergoes when the sentence is
    removed (redundancy-corrected via random masking of the rest of the
    message at label-collection time; see
    training.data.harvest_counterfactual_importance). Those expensive labels are
    distilled offline into a lightweight MiniLM + Ridge regressor loaded here;
    at inference, sentences scoring below tau are dropped. tau is the rate
    knob: sweeping it traces the rate-distortion frontier. Listener-aware by
    construction (labels were collected against the actual frozen receiver),
    but strictly extractive: it can delete, not re-encode.
    """

    name = "counterfactual"

    def __init__(
        self,
        tau: float = 0.5,
        model_path: str = "data/counterfactual_scorer.joblib",
    ):
        self.tau = tau
        self.model_path = model_path
        self._encoder = None
        self._regressor = None

    def _load(self):
        if self._regressor is None:
            import joblib
            from sentence_transformers import SentenceTransformer
            bundle = joblib.load(self.model_path)
            self._encoder = SentenceTransformer(bundle["encoder_name"], device="cpu")
            self._regressor = bundle["regressor"]
        return self._encoder, self._regressor

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        encoder, regressor = self._load()

        per_message: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message.append(sentences)
            all_sentences.extend(sentences)
        if not all_sentences:
            return messages

        embeddings = encoder.encode(all_sentences, normalize_embeddings=True)
        scores = regressor.predict(embeddings)

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message):
            kept = [
                s for s, score in zip(sentences, scores[cursor:cursor + len(sentences)])
                if score >= self.tau
            ]
            cursor += len(sentences)
            if kept:
                new_m = dict(m)
                new_m["content"] = " ".join(kept)
                result.append(new_m)
        return result


class RewriterCompressor(Compressor):
    """GRPO-trained abstractive rewriter (behavior-matching objective).

    A small policy LM compresses the whole view to a target fraction of its
    characters. Trained with reward = receiver-vote preservation + brevity
    - hallucination penalty (novel 4-grams absent from the source), so the
    policy is optimized to keep exactly the content the receiver's decision
    depends on without injecting its own reasoning.
    """

    name = "rewriter"

    _AGENT_LINE = re.compile(r"^Agent\s+(\d+)\s*:\s*(.*)$")

    def __init__(
        self,
        model_path: str = "data/rewriter_grpo/final",
        rate: float = 0.4,
        device: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.rate = rate
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        return self._model, self._tokenizer

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import torch

        model, tokenizer = self._load()

        view = "\n".join(
            f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
        )
        target = max(200, int(len(view) * self.rate))
        prompt = (
            "Compress the following group-discussion transcript to at most "
            f"{target} characters. Keep only decision-relevant facts and each "
            "agent's stated preference. Use ONLY information already present in "
            "the transcript - do not add reasoning or conclusions of your own. "
            "Keep the 'Agent N: ...' format.\n\n"
            f"Transcript:\n{view}\n\nCompressed transcript:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not completion:
            return messages

        # Parse back into message dicts; unattributed lines join the previous
        last_round = messages[-1]["round_num"]
        parsed: list[dict[str, Any]] = []
        for line in completion.splitlines():
            line = line.strip()
            if not line:
                continue
            m = self._AGENT_LINE.match(line)
            if m:
                parsed.append({
                    "agent_id": max(0, int(m.group(1)) - 1),
                    "round_num": last_round,
                    "content": m.group(2).strip(),
                })
            elif parsed:
                parsed[-1]["content"] += " " + line
        return parsed if parsed else messages


class BackrefCompressor(Compressor):
    """Stateful dedup that replaces repeats with agreement back-references.

    LZ77 at the semantic level: instead of silently deleting a sentence the
    receiver has already been shown (what stateful novelty dedup does), a
    cross-agent repeat collapses into a short marker naming the original
    speaker ("(restates a point made by Agent 2)"). Receivers get a fresh
    prompt every turn, so the marker cannot point back to text that is no
    longer in view; what it preserves is the agreement/reaffirmation signal
    that plain deletion discards. Self-repeats are dropped silently (they
    carry no agreement information).
    """

    name = "backref"

    def __init__(
        self,
        threshold: float = 0.85,
        drop_floor: float = 0.0,
        model_path: str = "data/selector_model.joblib",
    ):
        self.threshold = threshold
        # With drop_floor > 0, sentences the trained selector scores below the
        # floor are dropped silently (process chatter); dedup is unaffected.
        self.drop_floor = drop_floor
        self.model_path = model_path
        self._model = None
        self._classifier = None
        # receiver_id -> parallel lists: embeddings shown, source agent ids
        self._receiver_memory: dict[int, tuple[list[Any], list[int]]] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._receiver_memory = {}

    def _get_model(self):
        if self._model is None:
            if self.drop_floor > 0:
                import joblib
                from sentence_transformers import SentenceTransformer
                bundle = joblib.load(self.model_path)
                self._model = SentenceTransformer(bundle["encoder_name"], device="cpu")
                self._classifier = bundle["classifier"]
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
                )
        return self._model

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import numpy as np

        model = self._get_model()

        per_message_sentences: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message_sentences.append(sentences)
            all_sentences.extend(sentences)
        if not all_sentences:
            return messages

        embeddings = model.encode(all_sentences, normalize_embeddings=True)
        scores = None
        if self.drop_floor > 0 and self._classifier is not None:
            scores = self._classifier.predict_proba(embeddings)[:, 1]

        mem_embs, mem_agents = self._receiver_memory.get(receiver_id, ([], []))
        mem_embs, mem_agents = list(mem_embs), list(mem_agents)

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message_sentences):
            kept: list[str] = []
            echoed_agents: list[int] = []
            for k, (s, emb) in enumerate(
                    zip(sentences, embeddings[cursor:cursor + len(sentences)])):
                if scores is not None and float(scores[cursor + k]) < self.drop_floor:
                    continue
                if mem_embs:
                    sims = np.dot(np.stack(mem_embs), emb)
                    j = int(np.argmax(sims))
                    if float(sims[j]) >= self.threshold:
                        src = mem_agents[j]
                        if src != m["agent_id"] and src not in echoed_agents:
                            echoed_agents.append(src)
                        continue
                kept.append(s)
                mem_embs.append(emb)
                mem_agents.append(m["agent_id"])
            cursor += len(sentences)

            content = " ".join(kept)
            if echoed_agents:
                marker = "(restates points made by {})".format(
                    ", ".join(f"Agent {a + 1}" for a in sorted(echoed_agents))
                )
                content = f"{content} {marker}".strip()
            if content:
                new_m = dict(m)
                new_m["content"] = content
                result.append(new_m)

        self._receiver_memory[receiver_id] = (mem_embs, mem_agents)
        return result


_WORD_RE = re.compile(r"\S+")


class CodebookCompressor(Compressor):
    """Online shared-phrase dictionary (LZ78 at the phrase level), lossless.

    Multi-word phrases that recur in this task's traffic get short codes
    (§1, §2, ...). In each transmitted view, occurrences are replaced by the
    code and a one-line legend defining the codes actually used is prepended
    to the view, so the receiver can expand them exactly. Nothing is deleted,
    only re-encoded; phrases containing answer-option words are never encoded
    (votes must stay literal). Codes are only applied in views where they pay
    for their legend line (>= 2 occurrences).
    """

    name = "codebook"

    def __init__(
        self,
        min_words: int = 5,
        max_words: int = 10,
        min_count: int = 4,
        max_codes: int = 8,
        min_chars: int = 24,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_count = min_count
        self.max_codes = max_codes
        self.min_chars = min_chars
        self._counts: dict[str, int] = {}
        self._display: dict[str, str] = {}
        self._counted: set[tuple[int, int, int]] = set()
        self._codes: dict[str, int] = {}  # normalized phrase -> code number
        self._option_words: set[str] = set()

    def set_task_context(self, options: list[str]) -> None:
        self._counts = {}
        self._display = {}
        self._counted = set()
        self._codes = {}
        self._option_words = {
            w.lower() for option in options for w in option.split()
        }

    def _observe(self, messages: list[dict[str, Any]]) -> None:
        for m in messages:
            key = (m["agent_id"], m["round_num"], hash(m["content"]))
            if key in self._counted:
                continue
            self._counted.add(key)
            content = m["content"]
            words = [(w.group(0), w.start(), w.end()) for w in _WORD_RE.finditer(content)]
            for n in range(self.min_words, self.max_words + 1):
                for i in range(len(words) - n + 1):
                    surface = content[words[i][1]:words[i + n - 1][2]]
                    if len(surface) < self.min_chars:
                        continue
                    norm = " ".join(w[0].lower() for w in words[i:i + n])
                    if any(w[0].lower().strip('.,!?;:"()') in self._option_words
                           for w in words[i:i + n]):
                        continue
                    self._counts[norm] = self._counts.get(norm, 0) + 1
                    self._display.setdefault(norm, surface)

    def _promote(self) -> None:
        if len(self._codes) >= self.max_codes:
            return

        # Net chars saved if this phrase recurs as often as observed:
        # each substitution saves len-3 ("§k"), the legend line costs len+14.
        def gain(p: str) -> float:
            return self._counts[p] * (len(p) - 3) - (len(p) + 14)

        candidates = sorted(
            (p for p, c in self._counts.items()
             if c >= self.min_count and p not in self._codes and gain(p) > 50),
            key=lambda p: -gain(p),
        )
        for phrase in candidates:
            if len(self._codes) >= self.max_codes:
                break
            # Skip phrases nested inside an already-coded phrase
            if any(phrase in coded or coded in phrase for coded in self._codes):
                continue
            self._codes[phrase] = len(self._codes) + 1

    @staticmethod
    def _phrase_regex(norm: str) -> re.Pattern:
        return re.compile(
            r"\s+".join(re.escape(w) for w in norm.split()), re.IGNORECASE
        )

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        self._observe(messages)
        self._promote()
        if not self._codes:
            return messages

        contents = [m["content"] for m in messages]
        used: list[tuple[str, int]] = []
        # Longest phrases substitute first so shorter codes cannot break them
        for norm, code in sorted(self._codes.items(), key=lambda kv: -len(kv[0])):
            pattern = self._phrase_regex(norm)
            n_hits = sum(len(pattern.findall(c)) for c in contents)
            if n_hits < 2:
                continue
            contents = [pattern.sub(f"§{code}", c) for c in contents]
            used.append((norm, code))

        if not used:
            return messages

        legend = "[Shorthand used below: " + "; ".join(
            f'§{code} = "{self._display[norm]}"' for norm, code in
            sorted(used, key=lambda kv: kv[1])
        ) + "] "
        result = []
        for i, (m, content) in enumerate(zip(messages, contents)):
            new_m = dict(m)
            new_m["content"] = (legend + content) if i == 0 else content
            result.append(new_m)
        return result


class AdaptiveSelectCompressor(Compressor):
    """Learned dedup: the selector's relevance score sets how aggressively a
    sentence is treated as redundant, with no hard budget.

    Novel, relevant content is always kept (this is the fix for the budgeted
    selector's failure mode: a binding budget forces deletion of novel
    content). Low-relevance sentences face a permissive similarity threshold,
    so paraphrases of anything already shown get cut; sentences below
    drop_floor are cut outright (process chatter). Blind stateful dedup is
    the degenerate case thr_hi == thr_lo, drop_floor == 0.
    """

    name = "adaptive"

    def __init__(
        self,
        model_path: str = "data/selector_model.joblib",
        drop_floor: float = 0.10,
        thr_hi: float = 0.92,
        thr_lo: float = 0.60,
    ):
        self.model_path = model_path
        self.drop_floor = drop_floor
        self.thr_hi = thr_hi
        self.thr_lo = thr_lo
        self._encoder = None
        self._classifier = None
        self._receiver_memory: dict[int, list[Any]] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._receiver_memory = {}

    def _load(self):
        if self._classifier is None:
            import joblib
            from sentence_transformers import SentenceTransformer
            bundle = joblib.load(self.model_path)
            self._encoder = SentenceTransformer(bundle["encoder_name"], device="cpu")
            self._classifier = bundle["classifier"]
        return self._encoder, self._classifier

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import numpy as np

        encoder, classifier = self._load()

        per_message: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message.append(sentences)
            all_sentences.extend(sentences)
        if not all_sentences:
            return messages

        embeddings = encoder.encode(all_sentences, normalize_embeddings=True)
        scores = classifier.predict_proba(embeddings)[:, 1]

        memory = list(self._receiver_memory.get(receiver_id, []))
        keep_flags: list[bool] = []
        for emb, score in zip(embeddings, scores):
            if score < self.drop_floor:
                keep_flags.append(False)
                continue
            threshold = self.thr_lo + (self.thr_hi - self.thr_lo) * float(score)
            if memory:
                sims = np.dot(np.stack(memory), emb)
                if float(sims.max()) >= threshold:
                    keep_flags.append(False)
                    continue
            memory.append(emb)
            keep_flags.append(True)
        self._receiver_memory[receiver_id] = memory

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message):
            kept = [s for s, flag in
                    zip(sentences, keep_flags[cursor:cursor + len(sentences)]) if flag]
            cursor += len(sentences)
            if kept:
                new_m = dict(m)
                new_m["content"] = " ".join(kept)
                result.append(new_m)
        return result


class Gzip64Compressor(Compressor):
    """Strawman: classical byte-level compression of each message.

    Treats the channel as a byte pipe: zlib-compress each message and wrap in
    base64 so it survives as text. Fewer chars on the wire, but high-entropy
    base64 tokenizes ~4x denser than English (more tokens, not fewer) and the
    receiver cannot decode DEFLATE in-context. Included to measure, not to
    win.
    """

    name = "gzip64"

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        import base64
        import zlib

        result = []
        for m in messages:
            new_m = dict(m)
            payload = base64.b64encode(
                zlib.compress(m["content"].encode("utf-8"), 9)
            ).decode("ascii")
            new_m["content"] = f"[zlib+base64] {payload}"
            result.append(new_m)
        return result


class StackCompressor(Compressor):
    """Apply a pipeline of compressors in order, e.g. backref then codebook."""

    name = "stack"

    def __init__(self, stack: str = "backref,codebook"):
        names = [n.strip() for n in str(stack).split(",") if n.strip()]
        if not names:
            raise ValueError("stack compressor needs at least one child")
        self.children = [build_compressor(n) for n in names]

    def set_task_context(self, options: list[str]) -> None:
        for child in self.children:
            child.set_task_context(options)

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        for child in self.children:
            messages = child.compress(messages, receiver_id)
        return messages


class VIBSenderCompressor(Compressor):
    """Method 2: Variational Information-Bottleneck sender policy.

    A small policy LM (GRPO-trained, see training.programs.train_vib_grpo) re-encodes
    the view into a message M, rewarded during training on
    R(M) = E_probes[log q_phi(Y|primer,M)] + beta * log p_prior(M): a frozen
    receiver's forced-choice correctness on sampled true/false probes about
    the task's actual info items (content-preserving, not vote-preserving),
    plus a prior-model-codelength rate term. Unlike RewriterCompressor's
    vote-matching reward, this targets preserving the underlying facts
    regardless of what the group ultimately decides.
    """

    name = "vib_sender"

    def __init__(
        self,
        model_path: str = "data/vib_grpo/final",
        rate: float = 0.4,
        device: str | None = None,
        max_new_tokens: int = 400,
    ):
        self.model_path = model_path
        self.rate = rate
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        return self._model, self._tokenizer

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import torch

        model, tokenizer = self._load()

        view = "\n".join(
            f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
        )
        target = max(150, int(len(view) * self.rate))
        prompt = (
            "Compress the following group-discussion transcript into a short note "
            f"of at most {target} characters that preserves the key facts a "
            "teammate would need. Use ONLY information already present in the "
            "transcript - do not add reasoning or conclusions of your own.\n\n"
            f"Transcript:\n{view}\n\nCompressed note:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not completion:
            return messages

        last_round = messages[-1]["round_num"]
        return [{
            "agent_id": messages[-1]["agent_id"],
            "round_num": last_round,
            "content": completion,
        }]


class RepMatchSelectorCompressor(Compressor):
    """Extractive compression via a representational-match distilled scorer.

    Same shape as CounterfactualImportanceCompressor (MiniLM + regressor,
    tau threshold), but the label each sentence was trained on is the
    representational distance a frozen local proxy model's hidden state (at
    an intermediate layer) undergoes when the sentence is masked out of its
    message - not the noisy KL-divergence-over-answer-options label that
    Part 4 diagnosed as the weak link (val R^2 = 0.063). See
    training.data.build_repmatch_dataset for label collection against the
    representation server (training.services.repserver) and training.programs.train_selector
    (label-agnostic) for training. Purely extractive, no rep-server call at
    inference - the regressor was distilled offline.
    """

    name = "repmatch_selector"

    def __init__(
        self,
        tau: float = 0.5,
        model_path: str = "data/repmatch_selector.joblib",
    ):
        self.tau = tau
        self.model_path = model_path
        self._encoder = None
        self._regressor = None

    def _load(self):
        if self._regressor is None:
            import joblib
            from sentence_transformers import SentenceTransformer
            bundle = joblib.load(self.model_path)
            self._encoder = SentenceTransformer(bundle["encoder_name"], device="cpu")
            self._regressor = bundle["regressor"]
        return self._encoder, self._regressor

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        encoder, regressor = self._load()

        per_message: list[list[str]] = []
        all_sentences: list[str] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message.append(sentences)
            all_sentences.extend(sentences)
        if not all_sentences:
            return messages

        embeddings = encoder.encode(all_sentences, normalize_embeddings=True)
        scores = regressor.predict(embeddings)

        result = []
        cursor = 0
        for m, sentences in zip(messages, per_message):
            kept = [
                s for s, score in zip(sentences, scores[cursor:cursor + len(sentences)])
                if score >= self.tau
            ]
            cursor += len(sentences)
            if kept:
                new_m = dict(m)
                new_m["content"] = " ".join(kept)
                result.append(new_m)
        return result


class SaliencyPruneCompressor(Compressor):
    """Per-token pruning by gradient saliency toward representational self-preservation.

    Finer-grained than every sentence-level compressor in this file: for each
    message, queries the representation server's /saliency endpoint (one
    fixed-cost forward+backward pass through a frozen local proxy model, no
    growing state) for a per-token gradient-magnitude score, then greedily
    drops the lowest-saliency tokens until the message is under a hard
    character budget. Unlike LLMLingua-2's raw-perplexity pruning, saliency
    here is computed w.r.t. what shapes THIS message's own representation,
    not w.r.t. how surprising a token is to a generic LM.
    """

    name = "saliency"

    def __init__(
        self,
        rate: float = 0.4,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
    ):
        self.rate = rate
        self.repserver_url = repserver_url.rstrip("/")
        self.timeout = timeout

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import requests

        context = "\n".join(
            f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages[:-1]
        )

        result = []
        for i, m in enumerate(messages):
            text = m["content"]
            budget = max(1, int(len(text) * self.rate))
            if len(text) <= budget:
                result.append(m)
                continue

            local_context = context if i == len(messages) - 1 else ""
            try:
                resp = requests.post(
                    f"{self.repserver_url}/saliency",
                    json={"text": text, "context": local_context},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                tokens, scores = data["tokens"], data["scores"]
            except Exception:
                # Rep server unreachable: fail safe to the untouched message
                # rather than silently corrupting content.
                result.append(m)
                continue

            # Greedily keep highest-saliency tokens, in original order, under
            # the char budget (approximated via cumulative token text length).
            order = sorted(range(len(tokens)), key=lambda i: -scores[i])
            keep = [False] * len(tokens)
            used = 0
            for idx in order:
                piece = tokens[idx].replace("Ġ", " ").replace("▁", " ")
                cost = len(piece)
                if used + cost > budget and used > 0:
                    continue
                keep[idx] = True
                used += cost

            pieces = [
                tokens[i].replace("Ġ", " ").replace("▁", " ")
                for i in range(len(tokens)) if keep[i]
            ]
            new_content = "".join(pieces).strip()
            if new_content:
                new_m = dict(m)
                new_m["content"] = new_content
                result.append(new_m)
        return result


class RepMatchBestOfKCompressor(Compressor):
    """Training-free best-of-k selection by representational closeness.

    Generates a handful of cheap candidate compressions of the same view
    using compressors already in this registry (window at a couple of
    settings, novelty dedup, plain truncation), scores each candidate's
    representational distance to the ORIGINAL uncompressed view via the
    representation server, and deterministically keeps whichever candidate
    is closest. No gradient, no policy, nothing to reward-hack - a pure
    argmax over a fixed candidate set, so the only risk it carries is
    whatever risk the underlying candidate compressors already carry.
    """

    name = "repmatch_bestofk"

    def __init__(
        self,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
    ):
        self.repserver_url = repserver_url.rstrip("/")
        self.timeout = timeout
        self._candidates: list[Compressor] = [
            WindowCompressor(window_rounds=1),
            WindowCompressor(window_rounds=2),
            NoveltyCompressor(threshold=0.85, stateful=False),
            NoveltyCompressor(threshold=0.70, stateful=False),
        ]

    def _render(self, messages: list[dict[str, Any]]) -> str:
        return "\n".join(f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages)

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import numpy as np
        import requests

        # NOTE: the unmodified original is deliberately NOT in the candidate
        # pool - it would trivially have perfect self-similarity and always
        # "win," making this compressor a no-op by construction. It's used
        # only as the similarity TARGET below, never as something to select.
        candidates = [c.compress(list(messages), receiver_id) for c in self._candidates]
        # Only consider candidates that actually shrank the view - a
        # candidate identical (or larger) in char count contributes nothing
        # this method is meant to test.
        original_chars = sum(len(m["content"]) for m in messages)
        candidates = [
            c for c in candidates
            if c and sum(len(m["content"]) for m in c) < original_chars
        ]
        if not candidates:
            return messages

        texts = [self._render(messages)] + [self._render(c) for c in candidates]
        try:
            resp = requests.post(
                f"{self.repserver_url}/rep_batch",
                json={"items": [{"text": t} for t in texts]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            reps = np.array(resp.json()["reps"])
        except Exception:
            # Rep server unreachable: fail safe to the identity view.
            return messages

        original_rep = reps[0]
        best_idx, best_sim = 0, -1.0
        for i in range(len(candidates)):
            sim = float(np.dot(original_rep, reps[i + 1]))
            if sim > best_sim:
                best_sim, best_idx = sim, i
        return candidates[best_idx]


class RepMatchRewriterCompressor(Compressor):
    """GRPO-trained abstractive rewriter (representational-reconstruction objective).

    Same deployment shape as RewriterCompressor/VIBSenderCompressor (a small
    lazy-loaded local policy LM, generate-only, no rep-server dependency at
    inference), but trained with a materially different reward: cosine
    similarity between a frozen local proxy's hidden state when reading the
    policy's compressed output vs. reading the original text (see
    training.data.harvest_repmatch_data, training.programs.train_repmatch_grpo) -
    directly porting the NLAE recipe from
    agentic_learning_algorithms/trainers/nlae_stream_engine.py. Length is
    controlled by a HARD max_new_tokens cap during training and inference,
    deliberately not a soft codelength/log-prob rate term - the specific,
    named fix to the reward-hacking collapse that broke VIBSenderCompressor's
    codelength-as-rate design (Part 5).
    """

    name = "repmatch_rewriter"

    _AGENT_LINE = re.compile(r"^Agent\s+(\d+)\s*:\s*(.*)$")

    def __init__(
        self,
        model_path: str = "data/repmatch_grpo/final",
        rate: float = 0.4,
        device: str | None = None,
        max_new_tokens: int = 400,
    ):
        self.model_path = model_path
        self.rate = rate
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        return self._model, self._tokenizer

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import torch

        model, tokenizer = self._load()

        view = "\n".join(
            f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
        )
        target = max(200, int(len(view) * self.rate))
        prompt = (
            "Compress the following group-discussion transcript to at most "
            f"{target} characters so that a teammate reading only the "
            "compressed version would form the same understanding as reading "
            "the original. Use ONLY information already present in the "
            "transcript - do not add reasoning or conclusions of your own. "
            "Keep the 'Agent N: ...' format.\n\n"
            f"Transcript:\n{view}\n\nCompressed transcript:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not completion:
            return messages

        last_round = messages[-1]["round_num"]
        parsed: list[dict[str, Any]] = []
        for line in completion.splitlines():
            line = line.strip()
            if not line:
                continue
            m = self._AGENT_LINE.match(line)
            if m:
                parsed.append({
                    "agent_id": max(0, int(m.group(1)) - 1),
                    "round_num": last_round,
                    "content": m.group(2).strip(),
                })
            elif parsed:
                parsed[-1]["content"] += " " + line
        return parsed if parsed else messages


class TokenFilterCompressor(Compressor):
    """RL-trained token-level filtering policy (representational-match objective).

    A lightweight MLP head on frozen Qwen3-4B scores each token in the
    transmitted view for importance via its hidden state at the proxy
    model's target layer. Tokens with scores below the (1-tau) percentile
    are dropped. Higher tau = keep fewer tokens = more aggressive deletion.

    Trained via top-k REINFORCE with reward = max(0, -log(||h_comp - h_orig||^2 + eps))
    at multiple fixed keep rates (training.programs.train_tokenfilter_pg). Purely
    extractive: scores tokens, deletes low-scoring ones, never writes.

    At inference the full view is tokenized, scored, and the top k =
    (1 - tau) * N tokens are kept. The kept tokens are decoded and parsed
    back into per-agent message dicts.
    """

    name = "tokenfilter"

    _AGENT_LINE = re.compile(r"^Agent\s+(\d+)\s*:\s*(.*)$")

    def __init__(
        self,
        model_path: str = "data/tokenfilter_pg/final",
        tau: float = 0.5,
        device: str | None = None,
        max_length: int = 4096,
    ):
        self.model_path = model_path
        self.tau = tau
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._head = None

    def _load(self):
        if self._model is None:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from .token_filter_model import PolicyHead

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
            self._model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-4B",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            ).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

            hidden_dim = self._model.config.hidden_size
            self._head = PolicyHead(hidden_dim=hidden_dim).to(self.device)

            ckpt = torch.load(
                self.model_path + "/head_weights.pt", map_location=self.device, weights_only=True
            )
            self._head.load_state_dict(ckpt["head"])
            self._head.eval()
        return self._model, self._tokenizer, self._head

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        import torch
        import torch.nn.functional as F

        model, tokenizer, head = self._load()

        view = "\n".join(
            f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages
        )
        enc = tokenizer(
            view, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        ids = enc["input_ids"][0]

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[14][0]  # (seq_len, hidden_dim), bf16
            scores = head(h.float())  # (seq_len,) raw importance logits

        # tau controls the percentile cutoff: higher tau = more tokens deleted.
        # tau=0.5 means keep the top 50% of tokens by score.
        k = max(1, int(len(scores) * (1.0 - self.tau)))
        _, topk_idx = torch.topk(scores, k)
        keep = torch.zeros(len(scores), dtype=torch.bool, device=self.device)
        keep[topk_idx] = True

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        kept_ids = [int(tid) for tid, k_flag in zip(ids, keep) if k_flag and tid != bos_id and tid != eos_id]

        if not kept_ids:
            return messages

        compressed_text = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
        if not compressed_text:
            return messages

        last_round = messages[-1]["round_num"]
        parsed: list[dict[str, Any]] = []
        for line in compressed_text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = self._AGENT_LINE.match(line)
            if m:
                parsed.append({
                    "agent_id": max(0, int(m.group(1)) - 1),
                    "round_num": last_round,
                    "content": m.group(2).strip(),
                })
            elif parsed:
                parsed[-1]["content"] += " " + line
        return parsed if parsed else messages


class AutoencoderCompressor(Compressor):
    """Continuous-latent textual autoencoder with a learned projection bottleneck.

    Architecture:
      compress()  -- runs the model on text, extracts last-layer hidden
                     states, projects down through a learned bottleneck,
                     serializes as base64 within [AE]...[/AE].

      decompress() -- deserializes, projects back up through the learned
                      bottleneck, injects as prefix embeddings, and
                      generates the reconstructed text.

    The projection bottleneck (proj_down / proj_up) is trained end-to-end
    so gradients flow: reconstruction loss -> proj_up -> bottleneck ->
    proj_down -> hidden states -> encoder params.

    Transmitted size = num_latents × bottleneck_dim × fp16 bytes.
    """

    name = "autoencoder"

    _WRAPPER_RE = re.compile(r"^\[AE\](.*)\[/AE\]$")

    def __init__(
        self,
        model_path: str = "data/autoencoder/final",
        num_latents: int = 4,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.num_latents = num_latents
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._proj_down = None
        self._proj_up = None
        self._bottleneck_dim = None

    def _load(self):
        if self._model is None:
            import torch
            import torch.nn as nn
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
            ).to(self.device).eval()

            proj_path = os.path.join(self.model_path, "projection.pt")
            if os.path.exists(proj_path):
                ckpt = torch.load(proj_path, map_location=self.device, weights_only=True)
                self._bottleneck_dim = ckpt["bottleneck_dim"]
                H = self._model.config.hidden_size
                self._proj_down = nn.Linear(H, self._bottleneck_dim, dtype=torch.bfloat16).to(self.device)
                self._proj_up = nn.Linear(self._bottleneck_dim, H, dtype=torch.bfloat16).to(self.device)
                self._proj_down.load_state_dict(ckpt["proj_down"])
                self._proj_up.load_state_dict(ckpt["proj_up"])
                self._proj_down.eval()
                self._proj_up.eval()
        return self._model, self._tokenizer

    def _encode_one(self, text: str) -> str | None:
        import base64
        import numpy as np
        import torch

        model, tokenizer = self._load()
        enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        enc_ids = tokenizer.encode(enc_text, return_tensors="pt").to(self.device)
        seq_len = enc_ids.shape[1]
        last_layer = model.config.num_hidden_layers - 1

        with torch.no_grad():
            enc_out = model(enc_ids, output_hidden_states=True)
            hidden = enc_out.hidden_states[last_layer][0]

        if self.num_latents == 1:
            indices = [seq_len - 1]
        else:
            step = seq_len / self.num_latents
            indices = [min(int(i * step + step / 2), seq_len - 1)
                       for i in range(self.num_latents)]
        latents = hidden[indices]  # (num_latents, H)

        if self._proj_down is not None:
            latents = self._proj_down(latents).detach()  # (num_latents, bottleneck_dim)

        arr = latents.cpu().to(torch.float32).numpy()
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        return f"{self.num_latents}:{arr.shape[1]}:{b64}"

    def _decode_one(self, b64_data: str) -> str | None:
        import base64
        import numpy as np
        import torch

        model, tokenizer = self._load()

        parts = b64_data.split(":", 2)
        if len(parts) != 3:
            return None
        num_latents = int(parts[0])
        dim = int(parts[1])
        b64 = parts[2]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        latents = torch.from_numpy(arr.reshape(num_latents, dim)).to(
            self.device, dtype=torch.bfloat16
        )

        if self._proj_up is not None:
            latents = self._proj_up(latents)  # (num_latents, H)

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
                embeds[0, p] = latents[i]

        with torch.no_grad():
            past_kv = None
            current_emb = embeds
            generated_ids: list[int] = []
            for _ in range(self.max_new_tokens):
                out = self._model(
                    inputs_embeds=current_emb, past_key_values=past_kv, use_cache=True
                )
                past_kv = out.past_key_values
                next_id = out.logits[0, -1, :].argmax().item()
                if next_id == eos_id:
                    break
                generated_ids.append(next_id)
                current_emb = embed_layer(
                    torch.tensor([[next_id]], device=self.device)
                )
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True
        ).strip()
        return decoded if decoded else None

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for m in messages:
            latent = self._encode_one(m["content"])
            if latent is None:
                result.append(m)
            else:
                new_m = dict(m)
                new_m["content"] = f"[AE]{latent}[/AE]"
                result.append(new_m)
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            m2 = self._WRAPPER_RE.match(content)
            if m2:
                decoded = self._decode_one(m2.group(1))
                if decoded is not None:
                    new_m = dict(m)
                    new_m["content"] = decoded
                    result.append(new_m)
                    continue
            result.append(m)
        return result


class MWNOTAutoencoderCompressor(Compressor):
    """Continuous-latent textual autoencoder whose encoder is a multiwavelet
    neural-operator "generator" instead of K sampled hidden-state positions.

    Architecture:
      compress()  -- runs the model on text, takes ALL last-layer hidden
                     states (not just K sampled positions), multiscale-
                     decomposes and cross-attention-pools them (see
                     multimodal_comms.methods.autoencoders.mwnot_generator.SequenceGeneratorEncoder,
                     adapted from MWNOT-portable's adjacency-matrix neural
                     operator) into num_latents H-dim "generator" vectors,
                     serializes as base64 within [MWAE]...[/MWAE].

      decompress() -- deserializes, injects the generator vectors as prefix
                      embeddings, and generates the reconstructed text --
                      identical to AutoencoderCompressor's decode side.

    Requires a checkpoint produced by
    training.programs.pretrain_mwnot_autoencoder (base model + tokenizer +
    generator.pt). Transmitted size = num_latents x hidden_size x fp16
    bytes, same as AutoencoderCompressor with no bottleneck -- the gain
    here is meant to come from the encoder using the whole message instead
    of a handful of sampled positions, not from a narrower wire format.
    """

    name = "mwnot_autoencoder"

    _WRAPPER_RE = re.compile(r"^\[MWAE\](.*)\[/MWAE\]$")

    def __init__(
        self,
        model_path: str = "data/mwnot_autoencoder_pretrain/final",
        num_latents: int = 4,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.num_latents = num_latents
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._generator = None
        self._last_layer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from multimodal_comms.methods.autoencoders.mwnot_generator import SequenceGeneratorEncoder

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
            ).to(self.device).eval()
            self._last_layer = self._model.config.num_hidden_layers - 1

            gen_path = os.path.join(self.model_path, "generator.pt")
            ckpt = torch.load(gen_path, map_location=self.device, weights_only=True)
            self._generator = SequenceGeneratorEncoder(**ckpt["config"]).to(self.device)
            self._generator.load_state_dict(ckpt["state_dict"])
            self._generator.eval()
        return self._model, self._tokenizer

    def _encode_one(self, text: str) -> str | None:
        import base64
        import numpy as np
        import torch

        model, tokenizer = self._load()
        enc_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        enc_ids = tokenizer.encode(enc_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            enc_out = model(enc_ids, output_hidden_states=True)
            hidden = enc_out.hidden_states[self._last_layer]  # (1, S, H)
            valid_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=self.device)
            gen = self._generator(hidden.float(), valid_mask)[0].to(torch.bfloat16)  # (num_latents, H)

        arr = gen.cpu().to(torch.float32).numpy()
        b64 = base64.b64encode(arr.tobytes()).decode("ascii")
        return f"{self.num_latents}:{arr.shape[1]}:{b64}"

    def _decode_one(self, b64_data: str) -> str | None:
        import base64
        import numpy as np
        import torch

        model, tokenizer = self._load()

        parts = b64_data.split(":", 2)
        if len(parts) != 3:
            return None
        num_latents = int(parts[0])
        dim = int(parts[1])
        b64 = parts[2]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
        latents = torch.from_numpy(arr.reshape(num_latents, dim)).to(
            self.device, dtype=torch.bfloat16
        )

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
                embeds[0, p] = latents[i]

        with torch.no_grad():
            past_kv = None
            current_emb = embeds
            generated_ids: list[int] = []
            for _ in range(self.max_new_tokens):
                out = self._model(
                    inputs_embeds=current_emb, past_key_values=past_kv, use_cache=True
                )
                past_kv = out.past_key_values
                next_id = out.logits[0, -1, :].argmax().item()
                if next_id == eos_id:
                    break
                generated_ids.append(next_id)
                current_emb = embed_layer(
                    torch.tensor([[next_id]], device=self.device)
                )
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True
        ).strip()
        return decoded if decoded else None

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for m in messages:
            latent = self._encode_one(m["content"])
            if latent is None:
                result.append(m)
            else:
                new_m = dict(m)
                new_m["content"] = f"[MWAE]{latent}[/MWAE]"
                result.append(new_m)
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            m2 = self._WRAPPER_RE.match(content)
            if m2:
                decoded = self._decode_one(m2.group(1))
                if decoded is not None:
                    new_m = dict(m)
                    new_m["content"] = decoded
                    result.append(new_m)
                    continue
            result.append(m)
        return result


class GrammarCompressor(Compressor):
    """BPE grammar-based compression with a precomputed global codebook.

    Learns recurrent multi-word phrases across the discussion training corpus
    and maps them to compact symbols (§G0, §G1, ...). The codebook is
    shared a priori between sender and receiver.

    compress() replaces the longest-matching phrases with symbols (greedy,
    left-to-right longest-match). decompress() prepends a compact legend to
    the first message showing only the symbols USED in this view, so the
    receiver LLM can interpret them. Symbols not used in the view are not
    listed — overhead scales with active symbols, not codebook size.

    The grammar is lossless and deterministic.
    """

    name = "grammar"

    _SYMBOL_RE = re.compile(r"§\d+")

    def __init__(
        self,
        codebook_path: str = "data/grammar_codebook.json",
    ):
        self.codebook_path = codebook_path
        self._encode: dict[str, str] = {}  # phrase -> symbol
        self._decode: dict[str, str] = {}  # symbol -> phrase
        self._phrases_sorted: list[str] = []  # longest first for greedy match
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            codebook = json.loads(open(self.codebook_path).read())
        except (FileNotFoundError, json.JSONDecodeError):
            return

        for phrase, symbol in codebook.items():
            if phrase.strip():
                self._encode[phrase] = symbol
                self._decode[symbol] = phrase
        self._phrases_sorted = sorted(self._encode.keys(), key=len, reverse=True)

    def get_system_prompt_suffix(self) -> str:
        self._load()
        if not self._decode:
            return ""
        items = ", ".join(
            f"{s}={self._decode[s]}"
            for s in sorted(self._decode, key=lambda s: int(s[1:]))
        )
        return f"\n\nKey:{items}"

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        self._load()
        if not messages or not self._encode:
            return messages

        result = []
        for m in messages:
            text = m["content"]
            compressed = text
            for phrase in self._phrases_sorted:
                if phrase in compressed:
                    compressed = compressed.replace(phrase, self._encode[phrase])
            new_m = dict(m)
            new_m["content"] = compressed
            result.append(new_m)
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        return messages


def _render_view(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"Agent {m['agent_id'] + 1}: {m['content']}" for m in messages)


class CertifiedSpanDeletionCompressor(Compressor):
    """Greedy sentence-span deletion accepted by a representational certificate.

    Splits the view into spans (sentences), then makes a single left-to-right
    pass proposing to delete each remaining span. A deletion is kept only if
    the resulting view still satisfies

        cos(h(view_pruned), h(view_original)) >= 1 - eps

    against the frozen proxy exposed by the representation server (the same
    server backing repmatch_selector / repmatch_bestofk / saliency). Unlike
    token-level saliency pruning, whole sentences are removed, so the
    surviving text stays grammatical. The certificate converts the
    representational-match objective into a deterministic acceptance oracle
    with no trainable components: no RL, no reward hacking. Cost is bounded
    at exactly one certificate evaluation per span (one pass, no re-testing),
    each evaluation being a single batched /rep_batch call comparing the
    original view to the current candidate.

    Fails safe: if the representation server is unreachable, deletions
    already accepted are kept but no further deletion is attempted.
    """

    name = "certspan"

    def __init__(
        self,
        eps: float = 0.05,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
    ):
        self.eps = eps
        self.repserver_url = repserver_url
        self.timeout = timeout

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        original_text = _render_view(messages)

        per_message_sentences: list[list[str]] = []
        for m in messages:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(m["content"]) if s.strip()]
            per_message_sentences.append(sentences)
        alive = [[True] * len(s) for s in per_message_sentences]

        def render_candidate() -> list[dict[str, Any]]:
            out = []
            for m, sentences, flags in zip(messages, per_message_sentences, alive):
                kept = [s for s, f in zip(sentences, flags) if f]
                if kept:
                    new_m = dict(m)
                    new_m["content"] = " ".join(kept)
                    out.append(new_m)
            return out

        for mi, sentences in enumerate(per_message_sentences):
            for si in range(len(sentences)):
                alive[mi][si] = False
                candidate_text = _render_view(render_candidate())
                if not passes_certificate(
                    original_text, candidate_text, self.eps, self.repserver_url, self.timeout
                ):
                    alive[mi][si] = True  # revert: certificate rejected the deletion

        return render_candidate()


def _coerce_param(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _parse_ladder(spec: str) -> list["Compressor"]:
    """Parse 'name:key=val,key2=val2;name2;name3:key=val' into compressors."""
    compressors = []
    for item in (s.strip() for s in spec.split(";") if s.strip()):
        if ":" in item:
            name, kwargs_str = item.split(":", 1)
            kwargs = {}
            for pair in kwargs_str.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                k, v = pair.split("=", 1)
                kwargs[k.strip()] = _coerce_param(v.strip())
        else:
            name, kwargs = item, {}
        compressors.append(build_compressor(name.strip(), **kwargs))
    return compressors


class SemanticFallbackCompressor(Compressor):
    """Verification ladder: wraps any base compressors as a benchmark-general meta-method.

    Lossy compressors are strongly benchmark-dependent (a recipe that's a
    clear win on one task distribution can be a clear loss on another - the
    project-wide lesson behind `saliency`'s reversal between COMMA and
    HiddenBench). Given a ladder of compressors f_1, ..., f_{L-1} ordered
    from most to least aggressive, plus f_L = identity appended
    automatically, this transmits f_j(view) for the SMALLEST j (most
    aggressive) whose output still satisfies the certificate

        cos(h(f_j(view)), h(view)) >= 1 - eps

    Evaluation stops at the first passing rung, so cost is at most L
    certificate evaluations per view. Because f_L = identity always passes
    trivially (cos = 1), this bounds worst-case behavior at the identity
    channel while keeping the savings of aggressive compression wherever a
    rung's output is certified semantically safe.
    """

    name = "semfallback"

    DEFAULT_LADDER = (
        "window:window_rounds=1;"
        "novelty:threshold=0.75,stateful=True;"
        "backref:threshold=0.85;"
        "identity"
    )

    def __init__(
        self,
        ladder: str = DEFAULT_LADDER,
        eps: float = 0.10,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
    ):
        self.eps = eps
        self.repserver_url = repserver_url
        self.timeout = timeout
        self.ladder_spec = ladder
        self._children = _parse_ladder(ladder)
        if not self._children or not isinstance(self._children[-1], IdentityCompressor):
            self._children.append(IdentityCompressor())

    def set_task_context(self, options: list[str]) -> None:
        for child in self._children:
            child.set_task_context(options)

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        original_text = _render_view(messages)
        for child in self._children:
            candidate = child.compress(list(messages), receiver_id)
            candidate_text = _render_view(candidate) if candidate else ""
            if passes_certificate(
                original_text, candidate_text, self.eps, self.repserver_url, self.timeout
            ):
                return candidate
        return messages  # unreachable in practice: identity always certifies


class PredictiveDiffCompressor(Compressor):
    """Lossless diff code against a shared frozen LM (compress + decompress).

    LLM-generated text is highly predictable to LLMs, and inter-agent
    discussion is a best case for this: both endpoints already hold the
    shared conversation prefix, and cross-agent echo produces long stretches
    of predictable text. Sender and receiver hold the same small frozen LM
    p_theta (default Qwen2.5-0.5B-Instruct). For each message, a single
    teacher-forced forward pass checks, at every position, whether
    p_theta's greedy next-token prediction (given the true prefix) already
    matches the true token. Matching runs cost nothing to transmit; only
    mismatches ("corrections") are sent, as (run-length, token-id) pairs -
    precisely the bookkeeping of speculative decoding, reused here as a
    lossless codec rather than a sampling accelerant.

    The wire format is short text (`[PD]R12|C4521|R3...[/PD]`), not packed
    bits or base64 latents, so it avoids the token inflation that makes
    classical byte compression (see `Gzip64Compressor`) counterproductive in
    a token channel. decompress() replays p_theta deterministically -
    stepping token-by-token, splicing in each correction, letting the
    model's own greedy guess stand everywhere else - and reconstructs the
    original message exactly. Because reconstruction is exact, this cannot
    degrade task accuracy; its savings are largest precisely where
    discussion is most redundant (agreement, restatement, boilerplate).
    """

    name = "pdiff"

    _WRAPPER_RE = re.compile(r"^\[PD\](.*)\[/PD\]$", re.DOTALL)

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        max_context_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.max_context_tokens = max_context_tokens
        self._codec: PDiffCodec | None = None
        # (agent_id, round_num, content) -> wire string. encode_diff is a
        # pure function of (context, content), and under full_history every
        # historical message's context (everything before it) is fixed once
        # produced - so without this cache, a growing view under
        # full_history would redundantly re-run the frozen LM over every
        # old message on every single turn (quadratic in rounds).
        self._cache: dict[tuple, str] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._cache = {}

    def _get_codec(self) -> PDiffCodec:
        if self._codec is None:
            self._codec = PDiffCodec(
                model_path=self.model_path,
                device=self.device,
                max_context_tokens=self.max_context_tokens,
            )
        return self._codec

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        codec = self._get_codec()
        result = []
        for i, m in enumerate(messages):
            key = (m["agent_id"], m["round_num"], m["content"])
            cached = self._cache.get(key)
            if cached is not None:
                new_m = dict(m)
                new_m["content"] = cached
                result.append(new_m)
                continue
            context = _render_view(messages[:i])
            try:
                ops = codec.encode_diff(context, m["content"])
            except Exception:
                result.append(m)
                continue
            if not ops:
                result.append(m)
                continue
            wire = f"[PD]{PDiffCodec.serialize(ops)}[/PD]"
            self._cache[key] = wire
            new_m = dict(m)
            new_m["content"] = wire
            result.append(new_m)
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        codec = self._get_codec()
        result: list[dict[str, Any]] = []
        for m in messages:
            match = self._WRAPPER_RE.match(m["content"])
            if not match:
                result.append(m)
                continue
            context = _render_view(result)
            ops = PDiffCodec.deserialize(match.group(1))
            try:
                text = codec.replay(context, ops)
            except Exception:
                text = ""
            new_m = dict(m)
            new_m["content"] = text
            result.append(new_m)
        return result


class RateControlledDiffCompressor(PredictiveDiffCompressor):
    """Unified lossy<->lossless family: predictive-diff with omittable corrections.

    Extends `PredictiveDiffCompressor` with a lossy relaxation: a correction
    is omitted - letting the shared model's own greedy prediction stand -
    whenever dropping it still keeps the reconstructed message within the
    certificate's cosine budget of the original,

        cos(h(replay_without_this_correction), h(original)) >= 1 - eps.

    Corrections are tested in order, cumulatively (each accepted drop is
    folded in before testing the next), and dropped corrections are merged
    into the surrounding run before the wire format is serialized, so
    omissions genuinely shrink what's transmitted rather than just being
    ignored on the receiving end. eps = 0 collapses to the lossless code
    (only a byte-identical replay can satisfy cos >= 1, which no omission
    achieves), while increasing eps traces a rate-distortion curve measured
    in the receiver's representation space rather than lexical edit
    distance - one mechanism spanning the full spectrum instead of
    comparing unrelated compressor families.
    """

    name = "ratediff"

    def __init__(
        self,
        eps: float = 0.05,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        max_context_tokens: int = 2048,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
        max_corrections_to_search: int = 40,
    ):
        super().__init__(model_path=model_path, device=device, max_context_tokens=max_context_tokens)
        self.eps = eps
        self.repserver_url = repserver_url
        self.timeout = timeout
        self.max_corrections_to_search = max_corrections_to_search

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        codec = self._get_codec()
        result = []
        for i, m in enumerate(messages):
            content = m["content"]
            key = (m["agent_id"], m["round_num"], content)
            cached = self._cache.get(key)
            if cached is not None:
                new_m = dict(m)
                new_m["content"] = cached
                result.append(new_m)
                continue

            context = _render_view(messages[:i])
            try:
                ops = codec.encode_diff(context, content)
            except Exception:
                result.append(m)
                continue
            if not ops:
                result.append(m)
                continue

            n_corr = PDiffCodec.num_corrections(ops)
            keep_mask = [True] * n_corr
            if self.eps > 0 and 0 < n_corr <= self.max_corrections_to_search:
                for ci in range(n_corr):
                    trial_mask = list(keep_mask)
                    trial_mask[ci] = False
                    try:
                        candidate_text = codec.replay(context, ops, keep_mask=trial_mask)
                    except Exception:
                        continue
                    if passes_certificate(
                        content, candidate_text, self.eps, self.repserver_url, self.timeout
                    ):
                        keep_mask[ci] = False

            final_ops = PDiffCodec.apply_drops(ops, keep_mask) if n_corr else ops
            wire = f"[PD]{PDiffCodec.serialize(final_ops)}[/PD]"
            self._cache[key] = wire
            new_m = dict(m)
            new_m["content"] = wire
            result.append(new_m)
        return result


_FUNCTION_WORDS = {
    # articles
    "a", "an", "the",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "who", "whom", "whose", "which", "what",
    "whoever", "whatever", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "themselves",
    # be / have / do / modal auxiliaries
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could", "ought",
    # prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "over", "under", "again", "further", "then", "once", "out", "off",
    "near", "upon", "within", "without", "along", "across", "behind", "beyond",
    "plus", "except", "per", "via",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "although", "because", "since",
    "unless", "while", "whereas", "though", "if", "whether", "as",
    # other high-frequency function words
    "not", "no", "there", "here", "very", "too", "also", "just", "only", "own",
    "same", "than", "both", "each", "few", "more", "most", "other", "some", "such",
    "any", "all", "either", "neither",
}

_TOKEN_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


class TelegraphicCompressor(Compressor):
    """Function-word stripping with certified generative reinflation (compress + decompress).

    Closed-class function words are ~35-45% of English tokens and nearly
    perfectly recoverable from content words. The sender-side reduction is
    purely deterministic (a fixed function-word list, no model call): each
    message becomes content-word telegraphese. The receiver-side decoder -
    a small frozen LLM, default Qwen2.5-0.5B-Instruct - reinflates the
    telegraphic form into fluent text before the receiving agent ever sees
    it, which removes token-pruning's usual failure mode of grammatically
    fragmented input degrading downstream reasoning.

    Because reinflation is generative (not exact), the round trip is
    verified with the same certificate used elsewhere in this file: the
    reinflation is computed once, up front, at compress() time (so the
    accept/reject decision can be made before transmission), and the result
    is cached so decompress() doesn't have to regenerate it. Any message
    whose reinflation fails the certificate is transmitted unmodified
    (falls back to identity), never as unverified telegraphese.
    """

    name = "telegraphic"

    _WRAPPER_RE = re.compile(r"^\[TEL\](.*)\[/TEL\]$", re.DOTALL)

    def __init__(
        self,
        eps: float = 0.12,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 200,
        repserver_url: str = "http://127.0.0.1:8100",
        timeout: float = 30.0,
        min_chars: int = 40,
    ):
        self.eps = eps
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.repserver_url = repserver_url
        self.timeout = timeout
        self.min_chars = min_chars
        self._model = None
        self._tokenizer = None
        self._reinflation_cache: dict[str, str] = {}

    def set_task_context(self, options: list[str]) -> None:
        self._reinflation_cache = {}

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda:3" if torch.cuda.device_count() > 3 else (
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        return self._model, self._tokenizer

    @staticmethod
    def _reduce(text: str) -> str:
        kept = []
        for tok in text.split():
            core = _TOKEN_STRIP_RE.sub("", tok).lower()
            base = core[:-2] if core.endswith("'s") else core
            if core in _FUNCTION_WORDS or base in _FUNCTION_WORDS:
                continue
            kept.append(tok)
        return " ".join(kept)

    def _reinflate(self, telegraphic_text: str) -> str:
        import torch

        model, tokenizer = self._load()
        prompt = (
            "The following is a telegraphic note with function words removed. "
            "Rewrite it as fluent, grammatically complete text. Use ONLY the "
            "words/facts given - do not add new information or reasoning.\n\n"
            f"Telegraphic: {telegraphic_text}\n\nFluent version:"
        )
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        result = []
        for m in messages:
            content = m["content"]
            if len(content) < self.min_chars:
                result.append(m)
                continue
            telegraphic = self._reduce(content)
            if not telegraphic or len(telegraphic) >= len(content):
                result.append(m)
                continue
            try:
                reinflated = self._reinflate(telegraphic)
            except Exception:
                result.append(m)
                continue
            if not reinflated or not passes_certificate(
                content, reinflated, self.eps, self.repserver_url, self.timeout
            ):
                result.append(m)
                continue
            self._reinflation_cache[telegraphic] = reinflated
            new_m = dict(m)
            new_m["content"] = f"[TEL]{telegraphic}[/TEL]"
            result.append(new_m)
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages

        result = []
        for m in messages:
            match = self._WRAPPER_RE.match(m["content"])
            if not match:
                result.append(m)
                continue
            telegraphic = match.group(1)
            reinflated = self._reinflation_cache.get(telegraphic)
            if reinflated is None:
                try:
                    reinflated = self._reinflate(telegraphic)
                except Exception:
                    reinflated = telegraphic
            new_m = dict(m)
            new_m["content"] = reinflated
            result.append(new_m)
        return result


class SuperposeCompressor(Compressor):
    """Many messages -> ONE fixed-size latent packet via orthogonal binding.

    Every message in the receiver's view is encoded to (K, D) latents,
    bound with a slot-specific orthogonal key (slot = index within its
    chunk), and summed into a single superposed packet. The packet is
    serialized as [SPX]{header}|{packet}[/SPX], where the plaintext header
    carries only addressing metadata (agent_id, round_num per slot).

    decompress() unbinds each slot with its key and regenerates each
    message with the shared decoder LM. Views longer than max_slots are
    split into consecutive chunks of max_slots messages, one packet each --
    so max_slots is the superposition load, the main scaling dial.

    Transmitted size per packet = K x D x 4 bytes (fp32) + small header,
    independent of how many messages were superposed into it.

    Use a checkpoint fine-tuned with training.programs.pretrain_superpose; a
    vanilla autoencoder checkpoint decodes slot crosstalk poorly beyond
    max_slots=1.
    """

    name = "superpose"

    _WRAPPER_RE = re.compile(r"^\[SPX\](.*)\[/SPX\]$", re.DOTALL)

    def __init__(
        self,
        model_path: str = "data/autoencoder_pretrain/final",
        num_latents: int | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
        key_seed: int | None = None,
        key_mode: str | None = None,
        max_slots: int = 8,
    ):
        from multimodal_comms.methods.superposition.latent import LatentCodec, SuperposedPacketCodec

        # Key parameters must match how the checkpoint was trained; a
        # mismatched key_mode decodes silently to garbage. Default to the
        # checkpoint's own superpose_config.json and only fall back to the
        # historical qr/1234 when the checkpoint predates that record.
        sp_cfg_path = os.path.join(model_path, "superpose_config.json")
        sp_cfg = {}
        if os.path.exists(sp_cfg_path):
            try:
                sp_cfg = json.load(open(sp_cfg_path))
            except (OSError, json.JSONDecodeError):
                sp_cfg = {}
        if key_seed is None:
            key_seed = sp_cfg.get("key_seed", 1234)
        if key_mode is None:
            key_mode = sp_cfg.get("key_mode", "qr")

        self.max_slots = max_slots
        self._packet_codec = SuperposedPacketCodec(
            LatentCodec(
                model_path=model_path,
                num_latents=num_latents,
                device=device,
                max_new_tokens=max_new_tokens,
            ),
            key_seed=key_seed,
            key_mode=key_mode,
        )

    def compress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for start in range(0, len(messages), self.max_slots):
            chunk = messages[start:start + self.max_slots]
            texts_by_slot = {i: m["content"] for i, m in enumerate(chunk)}
            try:
                packet = self._packet_codec.encode_packet(texts_by_slot)
            except Exception:
                result.extend(chunk)
                continue
            header = json.dumps([
                {"slot": i, "agent_id": m["agent_id"], "round_num": m["round_num"]}
                for i, m in enumerate(chunk)
            ])
            result.append({
                "agent_id": chunk[-1]["agent_id"],
                "round_num": chunk[-1]["round_num"],
                "content": f"[SPX]{header}|{packet}[/SPX]",
            })
        return result

    def decompress(self, messages: list[dict[str, Any]], receiver_id: int) -> list[dict[str, Any]]:
        if not messages:
            return messages
        result: list[dict[str, Any]] = []
        for m in messages:
            match = self._WRAPPER_RE.match(m["content"])
            if not match:
                result.append(m)
                continue
            try:
                header_str, packet = match.group(1).split("|", 1)
                header = json.loads(header_str)
                for entry in header:
                    decoded = self._packet_codec.decode_slot(packet, entry["slot"])
                    result.append({
                        "agent_id": entry["agent_id"],
                        "round_num": entry["round_num"],
                        "content": decoded if decoded is not None else "",
                    })
            except Exception:
                result.append(m)
        return result


COMPRESSOR_REGISTRY = {
    "identity": IdentityCompressor,
    "window": WindowCompressor,
    "novelty": NoveltyCompressor,
    "llmlingua2": LLMLingua2Compressor,
    "learned": LearnedSelectCompressor,
    "rewriter": RewriterCompressor,
    "backref": BackrefCompressor,
    "codebook": CodebookCompressor,
    "adaptive": AdaptiveSelectCompressor,
    "gzip64": Gzip64Compressor,
    "stack": StackCompressor,
    "counterfactual": CounterfactualImportanceCompressor,
    "vib_sender": VIBSenderCompressor,
    "repmatch_selector": RepMatchSelectorCompressor,
    "saliency": SaliencyPruneCompressor,
    "repmatch_bestofk": RepMatchBestOfKCompressor,
    "repmatch_rewriter": RepMatchRewriterCompressor,
    "tokenfilter": TokenFilterCompressor,
    "autoencoder": AutoencoderCompressor,
    "mwnot_autoencoder": MWNOTAutoencoderCompressor,
    "grammar": GrammarCompressor,
    "certspan": CertifiedSpanDeletionCompressor,
    "semfallback": SemanticFallbackCompressor,
    "pdiff": PredictiveDiffCompressor,
    "telegraphic": TelegraphicCompressor,
    "ratediff": RateControlledDiffCompressor,
    "superpose": SuperposeCompressor,
}


def build_compressor(name: str, **params: Any) -> Compressor:
    name = (name or "identity").lower()
    if name not in COMPRESSOR_REGISTRY:
        available = ", ".join(COMPRESSOR_REGISTRY)
        raise ValueError(f"Unknown compressor: '{name}'. Available: {available}")
    return COMPRESSOR_REGISTRY[name](**params)
