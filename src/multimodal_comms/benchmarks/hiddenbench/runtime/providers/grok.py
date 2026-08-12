"""
Grok (xAI) LLM Provider.

Provides integration with xAI's Grok models via their API.
Grok uses an OpenAI-compatible API format.
"""

from typing import Any

from .base import LLMProvider, Message, Response, Role


class GrokProvider(LLMProvider):
    """
    Provider for xAI's Grok models.

    Grok uses an OpenAI-compatible API, so we use the openai package
    with a custom base URL.
    """

    AVAILABLE_MODELS = [
        "grok-beta",
        "grok-2",
        "grok-2-mini",
    ]

    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the Grok provider.

        Args:
            config: Configuration dictionary containing:
                - api_key: xAI API key (required)
                - default_model: Model to use (optional)
                - base_url: Custom API endpoint (optional)
                - timeout: Request timeout in seconds (optional)
        """
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazily initialize the Grok client (using OpenAI SDK)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for the Grok provider. "
                    "Install it with: pip install openai"
                )

            base_url = self.config.get("base_url", self.DEFAULT_BASE_URL)

            client_kwargs = {
                "api_key": self.config["api_key"],
                "base_url": base_url,
            }
            if "timeout" in self.config:
                client_kwargs["timeout"] = float(self.config["timeout"])

            self._client = OpenAI(**client_kwargs)

        return self._client

    def validate_config(self) -> bool:
        """Validate the Grok configuration."""
        if "api_key" not in self.config:
            raise ValueError("Grok provider requires 'api_key' in configuration")

        api_key = self.config["api_key"]
        if not api_key or api_key == "your-grok-api-key-here":
            raise ValueError(
                "Please set a valid xAI API key in your configuration. "
                "Get one from: https://console.x.ai/"
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
        Generate a completion using Grok.

        Args:
            messages: List of messages forming the conversation.
            model: Model to use (overrides default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional Grok-specific parameters.

        Returns:
            Response object containing the generated text.
        """
        client = self._get_client()
        model = model or self._model or "grok-beta"

        # Convert messages to OpenAI format (Grok uses same format)
        api_messages = []
        for msg in messages:
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

        # Add any extra kwargs
        for key, value in kwargs.items():
            if key not in request_kwargs:
                request_kwargs[key] = value

        # Make the API call
        response = client.chat.completions.create(**request_kwargs)

        # Extract the content
        choice = response.choices[0]
        content = choice.message.content or ""

        return Response(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available Grok models."""
        return self.AVAILABLE_MODELS.copy()
