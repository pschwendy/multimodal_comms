"""
OpenAI (GPT) LLM Provider.

Provides integration with OpenAI's GPT models via their API.
"""

from typing import Any

from .base import LLMProvider, Message, Response, Role


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's GPT models.

    Requires the 'openai' package to be installed.
    """

    AVAILABLE_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the OpenAI provider.

        Args:
            config: Configuration dictionary containing:
                - api_key: OpenAI API key (required)
                - default_model: Model to use (optional)
                - organization: Organization ID (optional)
                - base_url: Custom API endpoint (optional)
                - timeout: Request timeout in seconds (optional)
        """
        super().__init__(config)
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for the OpenAI provider. "
                    "Install it with: pip install openai"
                )

            client_kwargs = {
                "api_key": self.config["api_key"],
            }
            if "organization" in self.config:
                client_kwargs["organization"] = self.config["organization"]
            if "base_url" in self.config:
                client_kwargs["base_url"] = self.config["base_url"]
            if "timeout" in self.config:
                client_kwargs["timeout"] = float(self.config["timeout"])

            self._client = OpenAI(**client_kwargs)

        return self._client

    def validate_config(self) -> bool:
        """Validate the OpenAI configuration."""
        if "api_key" not in self.config:
            raise ValueError("OpenAI provider requires 'api_key' in configuration")

        api_key = self.config["api_key"]
        if not api_key or api_key == "your-openai-api-key-here":
            raise ValueError(
                "Please set a valid OpenAI API key in your configuration. "
                "Get one from: https://platform.openai.com/api-keys"
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
        Generate a completion using GPT.

        Args:
            messages: List of messages forming the conversation.
            model: Model to use (overrides default).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            Response object containing the generated text.
        """
        client = self._get_client()
        model = model or self._model or "gpt-4o"

        # Convert messages to OpenAI format
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

        # Pass through arbitrary vendor-specific fields (e.g. chat_template_kwargs
        # for OpenAI-compatible servers like vLLM) configured under the provider's
        # 'extra_body' config key.
        if "extra_body" in self.config:
            request_kwargs["extra_body"] = self.config["extra_body"]

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
        """Get list of available GPT models."""
        return self.AVAILABLE_MODELS.copy()
