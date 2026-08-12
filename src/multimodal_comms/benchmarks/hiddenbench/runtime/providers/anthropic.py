"""
Anthropic (Claude) LLM Provider.

Provides integration with Anthropic's Claude models via their API.
"""

from typing import Any

from .base import LLMProvider, Message, Response, Role


class AnthropicProvider(LLMProvider):
    """
    Provider for Anthropic's Claude models.

    Requires the 'anthropic' package to be installed.
    """

    # Current models (Claude 4.x family)
    AVAILABLE_MODELS = [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        # Legacy models (still available)
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-haiku-20240307",  # Training cutoff: Aug 2023 (predates HiddenBench paper)
    ]

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the Anthropic provider.

        Args:
            config: Configuration dictionary containing:
                - api_key: Anthropic API key (required)
                - default_model: Model to use (optional)
                - base_url: Custom API endpoint (optional)
                - timeout: Request timeout in seconds (optional)
        """
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for the Anthropic provider. "
                    "Install it with: pip install anthropic"
                )

            client_kwargs = {
                "api_key": self.config["api_key"],
            }
            if "base_url" in self.config:
                client_kwargs["base_url"] = self.config["base_url"]
            if "timeout" in self.config:
                client_kwargs["timeout"] = float(self.config["timeout"])

            self._client = anthropic.Anthropic(**client_kwargs)

        return self._client

    def validate_config(self) -> bool:
        """Validate the Anthropic configuration."""
        if "api_key" not in self.config:
            raise ValueError("Anthropic provider requires 'api_key' in configuration")

        api_key = self.config["api_key"]
        if not api_key or api_key == "your-anthropic-api-key-here":
            raise ValueError(
                "Please set a valid Anthropic API key in your configuration. "
                "Get one from: https://console.anthropic.com/"
            )

        return True

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
        Generate a completion using Claude.

        Args:
            messages: List of messages forming the conversation.
            model: Model to use (overrides default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            Response object containing the generated text.
        """
        client = self._get_client()
        model = model or self._model or "claude-sonnet-4-5-20250929"

        # Extract system message if present
        system_content = None
        api_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Build API request
        request_kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system_content:
            request_kwargs["system"] = system_content

        # Add any extra kwargs
        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        # Make the API call
        response = client.messages.create(**request_kwargs)

        # Extract the text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return Response(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Claude models."""
        return self.AVAILABLE_MODELS.copy()
