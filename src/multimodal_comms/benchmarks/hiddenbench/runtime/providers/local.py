"""
Local Llama LLM Provider.

Provides integration with locally-hosted open-weight models using llama-cpp-python.
This is ideal for running on Apple Silicon Macs with Metal acceleration.
"""

from typing import Any

from .base import LLMProvider, Message, Response, Role


class LocalLlamaProvider(LLMProvider):
    """
    Provider for locally-hosted Llama models using llama-cpp-python.

    Requires the 'llama-cpp-python' package to be installed.
    For best performance on Mac, install with Metal support:
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

    Recommended models (download GGUF format from HuggingFace):
    - Llama 3.2 3B: Good balance of speed and capability
    - Llama 3.1 8B: Better reasoning, needs ~8GB RAM
    - Mistral 7B: Good alternative
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the local Llama provider.

        Args:
            config: Configuration dictionary containing:
                - model_path: Path to the GGUF model file (required)
                - n_gpu_layers: Number of layers to offload to GPU (-1 for all)
                - n_ctx: Context window size (default: 4096)
                - n_threads: Number of CPU threads (0 for auto)
                - verbose: Enable verbose logging (default: False)
        """
        super().__init__(config)
        self._llm = None
        self._model_path = config.get("model_path")

    def _get_llm(self):
        """Lazily initialize the Llama model."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "The 'llama-cpp-python' package is required for the local provider. "
                    "Install it with: pip install llama-cpp-python\n"
                    "For Mac with Metal GPU acceleration:\n"
                    "  CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python"
                )

            if not self._model_path:
                raise ValueError(
                    "Local provider requires 'model_path' in configuration. "
                    "Please specify the path to your GGUF model file."
                )

            # Get configuration options
            n_gpu_layers = self.config.get("n_gpu_layers", -1)
            n_ctx = self.config.get("n_ctx", 4096)
            n_threads = self.config.get("n_threads", 0)
            verbose = self.config.get("verbose", False)

            # If n_threads is 0, let llama.cpp auto-detect
            llm_kwargs = {
                "model_path": self._model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "verbose": verbose,
            }
            if n_threads > 0:
                llm_kwargs["n_threads"] = n_threads

            self._llm = Llama(**llm_kwargs)

        return self._llm

    def validate_config(self) -> bool:
        """Validate the local Llama configuration."""
        if "model_path" not in self.config:
            raise ValueError("Local provider requires 'model_path' in configuration")

        model_path = self.config["model_path"]
        if not model_path or model_path == "/path/to/your/model.gguf":
            raise ValueError(
                "Please set a valid model path in your configuration. "
                "Download a GGUF model from HuggingFace and update the path."
            )

        # Check if file exists
        import os
        if not os.path.exists(model_path):
            raise ValueError(
                f"Model file not found: {model_path}\n"
                "Please download a GGUF model file from HuggingFace."
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
        Generate a completion using the local Llama model.

        Args:
            messages: List of messages forming the conversation.
            model: Ignored for local provider (uses configured model).
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional llama.cpp parameters.

        Returns:
            Response object containing the generated text.
        """
        llm = self._get_llm()

        # Convert messages to chat format
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg.role.value,
                "content": msg.content,
            })

        # Generate response
        response = llm.create_chat_completion(
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract content
        choice = response["choices"][0]
        content = choice["message"]["content"] or ""

        # Get usage info
        usage = response.get("usage", {})

        return Response(
            content=content,
            model=self._model_path or "local",
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            finish_reason=choice.get("finish_reason"),
            raw_response=response,
        )

    def get_available_models(self) -> list[str]:
        """Get list of available models (just the configured one)."""
        if self._model_path:
            return [self._model_path]
        return []

    @property
    def model(self) -> str | None:
        """Return the model path."""
        return self._model_path

    @model.setter
    def model(self, value: str) -> None:
        """Set the model path (requires re-initialization)."""
        if value != self._model_path:
            self._model_path = value
            self._llm = None  # Force re-initialization
