"""
Base LLM Provider interface.

All LLM providers must implement this interface to be usable with HiddenBench.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Role(Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content, metadata=metadata)

    @classmethod
    def user(cls, content: str, **metadata: Any) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content, metadata=metadata)

    @classmethod
    def assistant(cls, content: str, **metadata: Any) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, metadata=metadata)


@dataclass
class Response:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement the `complete` method to generate responses
    from the LLM given a list of messages.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration dictionary.
        """
        self.config = config
        self._model = config.get("default_model")

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__.replace("Provider", "").lower()

    @property
    def model(self) -> str | None:
        """Return the current model."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Set the model to use."""
        self._model = value

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs: Any,
    ) -> Response:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of messages forming the conversation.
            model: Model to use (overrides default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Response object containing the generated text.
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the provider configuration.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    def get_available_models(self) -> list[str]:
        """
        Get list of available models for this provider.

        Returns:
            List of model identifiers.
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
