"""Benchmark-neutral multi-agent communication algorithms."""

from .core import CommunicationMethod, Message, MethodContext, Transmission
from .registry import create_method, get_method_spec, list_methods

__all__ = [
    "CommunicationMethod",
    "Message",
    "MethodContext",
    "Transmission",
    "create_method",
    "get_method_spec",
    "list_methods",
]

__version__ = "0.1.0"
