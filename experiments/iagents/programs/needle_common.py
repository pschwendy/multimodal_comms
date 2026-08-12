"""Shared helpers for the Needle_in_the_Persona offline compression eval.

This module is the glue between the iAgents two-agent (instructor/assistant)
communication stack and the shared hiddenbench compression `channel` library.

Role mapping for a Needle record (identical for 1hop and 2hop):
  - instructor master = "charlie"
      private "other" chat  = modified_charlie_dave_conversation
        (1hop: pure distractor; 2hop: holds Dave's half of the needle)
  - assistant  master = "bob"
      private "other" chat  = modified_alice_bob_conversation
        (holds Alice's half / the whole 1hop needle)
  - shared "current" chat between them = chat_bob_charlie
  - task  = task_prompt   answer = answer

The needle can only be resolved by the two agents actually communicating, so
what the *channel* transmits between them is exactly what compression bites on.
"""

import json
import os
import re
import sys

# --- make iAgents + the shared hiddenbench library importable ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_IAGENTS = os.path.dirname(_HERE)
_HIDDENBENCH_SRC = os.path.abspath(
    os.path.join(_IAGENTS, "..", "hiddenbench", "src")
)

from multimodal_comms.apps.iagents.runtime.agent import ThinkAgent  # noqa: E402


# --------------------------------------------------------------------------
# Record loading
# --------------------------------------------------------------------------
def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def record_roles(rec):
    """Return the (instructor, assistant) seeding for one record."""
    return {
        "instructor_master": "charlie",
        "assistant_master": "bob",
        "instructor_other": rec.get("modified_charlie_dave_conversation", ""),
        "assistant_other": rec.get("modified_alice_bob_conversation", ""),
        "current_chat": rec.get("chat_bob_charlie", ""),
        "task": rec.get("task_prompt", ""),
        "answer": rec.get("answer", ""),
    }


# --------------------------------------------------------------------------
# Agent whose private context is injected in-memory (no DB dependency).
# --------------------------------------------------------------------------
class OfflineThinkAgent(ThinkAgent):
    """ThinkAgent whose current/other chat history is seeded directly from a
    Needle record instead of being retrieved from the chat database."""

    def __init__(self, master, backend, task, current_chat, other_chat,
                 is_assistant=False):
        super().__init__(master, backend, task, is_assistant)
        self._current_chat = current_chat or ""
        self._other_chat = other_chat or ""

    def get_current_chat_history(self, receiver, communication_history):
        return "\n" + self._current_chat + "\n"

    def get_other_chat_history(self, receiver, communication_history=None):
        return "\n" + self._other_chat + "\n"

    def assemble_prompt(self, receiver, communication_history,
                        current_chat_history, other_chat_history):
        # Same as ThinkAgent.assemble_prompt but without the DB profile lookup.
        system_prompt = "\n".join([
            "\n".join(self.system_prompt['role']).format(
                master=self.master, contact=receiver),
            "\n".join(self.system_prompt['chat_history']).format(
                master=self.master, contact=receiver,
                current_chat_history=current_chat_history,
                other_chat_history=other_chat_history),
            "\n".join(self.system_prompt['task']).format(
                contact=receiver, task=self.task),
            "\n".join(self.system_prompt['agent_chat_history']).format(
                contact=receiver,
                agent_chat_history="\n".join(communication_history),
                master=self.master),
            "\n".join(self.system_prompt['return_format_withinfonav']).format(
                infonav=self.infonav_plan,
                unknown_facts=self.mindfill_tool.get_unknown_facts()),
        ])
        return system_prompt


# --------------------------------------------------------------------------
# Grader: normalized string match (modeled on FriendsTV make_qa_dataset.norm)
# --------------------------------------------------------------------------
_STOP = set(
    "a an the is are was were be been being of to and or but in on at for with "
    "about as by from this that these those it its i you he she they we my your "
    "her his their our do does did have has had will would can could not no yes "
    "s t re ve ll m d".split()
)


def _norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())


def grade(conclusion, answer, recall_threshold=0.6):
    """True if the ground-truth answer is recoverable from the conclusion.

    Compact-substring for short answers; content-word recall for long
    'fact'-style answers.
    """
    nc = _norm(conclusion)
    na = _norm(answer)
    na_compact = na.replace(" ", "")
    nc_compact = nc.replace(" ", "")
    if na_compact and na_compact in nc_compact:
        return True
    ans_words = [w for w in na.split() if w and w not in _STOP]
    if not ans_words:
        return bool(na_compact) and na_compact in nc_compact
    concl_words = set(nc.split())
    recall = sum(1 for w in ans_words if w in concl_words) / len(ans_words)
    return recall >= recall_threshold
