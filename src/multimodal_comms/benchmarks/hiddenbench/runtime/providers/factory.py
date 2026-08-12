"""
Provider factory for creating LLM provider instances.
"""

from typing import Any

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .grok import GrokProvider
from .local import LocalLlamaProvider


# Registry of available providers
PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "grok": GrokProvider,
    "local": LocalLlamaProvider,
}


def get_available_providers() -> list[str]:
    """
    Get list of available provider names.

    Returns:
        List of provider identifiers.
    """
    return list(PROVIDER_REGISTRY.keys())


def create_provider(provider_name: str, config: dict[str, Any]) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider_name: Name of the provider (e.g., "anthropic", "openai").
        config: Provider-specific configuration dictionary.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    provider_name = provider_name.lower()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(get_available_providers())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Available providers: {available}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(config)


def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """
    Register a custom provider.

    Args:
        name: Name to register the provider under.
        provider_class: Provider class to register.
    """
    PROVIDER_REGISTRY[name.lower()] = provider_class
