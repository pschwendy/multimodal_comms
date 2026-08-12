"""Per-receiver communication-compression middleware for COMMA.

Monkeypatches ``main.GameManager`` so that instead of broadcasting every raw
message dict identically to each agent's ``.conversation``, the manager keeps a
canonical chronological transcript of the Expert<->Solver channel and, for each
recipient, appends a *compressed view* of that transcript computed via the
shared hiddenbench compressor library.

Message flow in COMMA (confirmed by reading main.py / agent.py):
  * Solver speaks  -> Agent.step -> GameManager.add_to_conversations({"from":"SOLVER",...})   (broadcast)
  * Expert speaks  -> conversation_loop -> GameManager.add_to_conversations({"from":"EXPERT",...}) (broadcast)
  * Env feedback   -> Module.execute_action -> add_to_conversations({"from":"ENVIRONMENT",...}, role="solver")
So ``add_to_conversations`` is the single common funnel for both roles.

Canonical transcript entries use the hiddenbench shape
``{"agent_id": int, "round_num": int, "content": str}`` with SOLVER=0, EXPERT=1.
Only SOLVER/EXPERT messages are compressible; ENVIRONMENT feedback is private,
short and left untouched (kept per-agent and re-appended after the compressed
channel view). round_num increments once per full Solver+Expert cycle.

Activated by importing this module and calling ``activate()`` (done by
run_comma.py). Compressor is selected via env vars:
  COMMA_COMPRESSOR         compressor name (default "identity")
  COMMA_COMPRESSOR_PARAMS  optional JSON dict of kwargs for build_compressor
"""

import os
import sys
import json

# HiddenBench compatibility types now live in this monorepo namespace.
_HB_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hiddenbench", "src")

from multimodal_comms.benchmarks.hiddenbench.runtime.channel import build_compressor  # noqa: E402


def _agent_id(role: str) -> int:
    return 0 if role.lower().startswith("solver") else 1


def _from_label(agent_id: int) -> str:
    return "SOLVER" if agent_id == 0 else "EXPERT"


def activate(compressor_name=None, compressor_params=None):
    """Patch GameManager to route the channel through a compressor."""
    import multimodal_comms.apps.comma.main as main

    if compressor_name is None:
        compressor_name = os.getenv("COMMA_COMPRESSOR", "identity")
    if compressor_params is None:
        raw = os.getenv("COMMA_COMPRESSOR_PARAMS", "")
        compressor_params = json.loads(raw) if raw.strip() else {}

    GM = main.GameManager

    def _ensure_state(self):
        if not hasattr(self, "_ca_channel"):
            self._ca_channel = []          # canonical [{agent_id,round_num,content}]
            self._ca_round = 0             # increments per Solver+Expert cycle
            self._ca_env = {}              # id(agent) -> [raw env dicts]
            self._ca_cache = {}            # receiver_id -> (channel_len, compressed_conv)
            self._ca_compressor = build_compressor(compressor_name, **compressor_params)
            self._ca_name = compressor_name

    def _rebuild(self):
        # Cache the compressed channel per receiver, keyed by channel length, so
        # env-feedback messages (which do not change the channel) never trigger a
        # recompress. Heavy compressors (rewriter/saliency/bestofk) are thus only
        # invoked ~once per new Solver/Expert message per receiver.
        clen = len(self._ca_channel)
        for agent in self.agents:
            rid = _agent_id(agent.role)
            cached = self._ca_cache.get(rid)
            if cached is not None and cached[0] == clen:
                conv = list(cached[1])
            else:
                view = [dict(m) for m in self._ca_channel]
                try:
                    compressed = self._ca_compressor.compress(view, rid)
                except Exception as e:
                    print(f"[channel_adapter] compress failed ({self._ca_name}): {e}; falling back to identity view")
                    compressed = view
                # Never hand back an empty channel when there was content.
                if view and not compressed:
                    compressed = [view[-1]]
                conv = [
                    {"from": _from_label(m["agent_id"]), "value": m["content"]}
                    for m in compressed
                ]
                self._ca_cache[rid] = (clen, list(conv))
            conv = conv + self._ca_env.get(id(agent), [])
            agent.conversation = conv

    def patched_add_to_conversations(self, data, role=None):
        _ensure_state(self)
        frm = str(data.get("from", "")).lower()
        if frm.startswith("solver") or frm.startswith("expert"):
            aid = _agent_id(frm)
            self._ca_channel.append({
                "agent_id": aid,
                "round_num": self._ca_round,
                "content": str(data.get("value", "")),
            })
            if frm.startswith("expert"):
                self._ca_round += 1
            _rebuild(self)
        else:
            # Private environment feedback (or anything non-channel): keep raw,
            # respecting the optional role filter, and re-append on rebuild.
            for agent in self.agents:
                if role and not agent.role.lower().startswith(role):
                    continue
                self._ca_env.setdefault(id(agent), []).append(dict(data))
            _rebuild(self)

    # Reset canonical transcript at each new puzzle. advance_puzzle is called
    # between puzzles; setup_puzzle is called once at start and once per advance.
    _orig_setup = GM.setup_puzzle

    def patched_setup_puzzle(self):
        self._ca_channel = []
        self._ca_round = 0
        self._ca_env = {}
        self._ca_cache = {}
        # keep a single compressor instance for the whole run (stateful ones
        # reset via their own set_task_context if needed)
        if not hasattr(self, "_ca_compressor"):
            self._ca_compressor = build_compressor(compressor_name, **compressor_params)
            self._ca_name = compressor_name
        # per-task reset hook for stateful compressors
        try:
            self._ca_compressor.set_task_context([])
        except Exception:
            pass
        return _orig_setup(self)

    GM.add_to_conversations = patched_add_to_conversations
    GM.setup_puzzle = patched_setup_puzzle
    print(f"[channel_adapter] activated compressor='{compressor_name}' params={compressor_params}")
