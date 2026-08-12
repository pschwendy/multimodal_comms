"""
HiddenBench: A benchmark for evaluating collective reasoning in multi-agent LLM systems.

Based on the Hidden Profile paradigm from social psychology, this benchmark assesses
whether groups of language models can successfully integrate distributed information
when each agent holds asymmetric knowledge pieces.
"""

__version__ = "0.1.0"
__author__ = "HiddenBench Contributors"

from .benchmark import HiddenBench
from .config import Config
from .providers.base import LLMProvider
from .task import Task, TaskResult

__all__ = [
    "HiddenBench",
    "Config",
    "LLMProvider",
    "Task",
    "TaskResult",
]
