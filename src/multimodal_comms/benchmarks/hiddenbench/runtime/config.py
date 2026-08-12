"""
Configuration management for HiddenBench.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    api_key: str = ""
    default_model: str = ""
    base_url: str | None = None
    timeout: float | None = None
    # Token estimation parameters for this provider
    # These help estimate total token usage before running the benchmark
    est_base_input_tokens: int = 800  # Base input tokens per call (scenario + prompt)
    est_output_tokens: int = 250  # Average output tokens per response
    est_context_growth: int = 300  # Tokens added to context per discussion message
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for provider initialization."""
        d = {"api_key": self.api_key}
        if self.default_model:
            d["default_model"] = self.default_model
        if self.base_url:
            d["base_url"] = self.base_url
        if self.timeout:
            d["timeout"] = self.timeout
        d.update(self.extra)
        return d


def _get_default_data_dir() -> str:
    """Get the default data directory (relative to package location)."""
    # Try to find the package's data directory
    benchmark_dir = Path(__file__).parents[1]
    data_dir = benchmark_dir / "data" / "hiddenbench_official"
    if data_dir.exists():
        return str(data_dir)
    # Fallback to current directory's data folder
    return str(data_dir)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    provider: str = "anthropic"
    model: str | None = None
    num_agents: int = 4
    num_rounds: int = 15
    temperature: float = 0.7
    max_tokens: int = 500
    # Primary data source - official HiddenBench data
    data_dir: str = field(default_factory=_get_default_data_dir)
    # Additional custom tasks directory
    tasks_dir: str = "./tasks"
    reports_dir: str = "./reports"
    run_full_profile: bool = True
    seed: int | None = None
    # Which data sources to use
    use_official_data: bool = True
    use_custom_tasks: bool = True


@dataclass
class ChannelConfig:
    """Configuration for the inter-agent communication channel.

    Defaults reproduce the original HiddenBench protocol exactly.
    """
    # "full_history" (original: retransmit entire discussion each turn)
    # or "delta" (each agent only receives messages it has not seen yet)
    protocol: str = "full_history"
    # identity | window | novelty | llmlingua2
    compressor: str = "identity"
    # Sender-side message style: default | concise | schema
    message_style: str = "default"
    # Compressor parameters
    window_rounds: int = 2
    novelty_threshold: float = 0.85
    novelty_stateful: bool = False
    lingua_rate: float = 0.4
    selector_rate: float = 0.5
    selector_dedup: bool = False
    selector_model: str = "data/selector_model.joblib"
    rewriter_rate: float = 0.4
    rewriter_model: str = "data/rewriter_grpo/final"
    backref_threshold: float = 0.85
    backref_drop_floor: float = 0.0
    codebook_min_count: int = 3
    adaptive_drop_floor: float = 0.10
    adaptive_thr_hi: float = 0.92
    adaptive_thr_lo: float = 0.60
    stack: str = "backref,codebook"
    counterfactual_tau: float = 0.5
    counterfactual_model: str = "data/counterfactual_scorer.joblib"
    vib_rate: float = 0.4
    vib_model: str = "data/vib_grpo/final"
    repmatch_tau: float = 0.5
    repmatch_selector_model: str = "data/repmatch_selector.joblib"
    saliency_rate: float = 0.4
    repserver_url: str = "http://127.0.0.1:8100"
    repmatch_rewriter_rate: float = 0.4
    repmatch_rewriter_model: str = "data/repmatch_grpo/final"
    tokenfilter_tau: float = 0.5
    tokenfilter_model: str = "data/tokenfilter_pg/final"
    tokenfilter_device: str = "cuda:3"
    autoencoder_model: str = "data/autoencoder/final"
    autoencoder_num_latents: int = 4
    autoencoder_device: str = "cuda:3"
    mwnot_autoencoder_model: str = "data/mwnot_autoencoder_pretrain/final"
    mwnot_autoencoder_num_latents: int = 4
    mwnot_autoencoder_device: str = "cuda:3"
    grammar_codebook: str = "data/grammar_codebook.json"
    certspan_eps: float = 0.05
    semfallback_ladder: str = (
        "window:window_rounds=1;"
        "novelty:threshold=0.75,stateful=True;"
        "backref:threshold=0.85;"
        "identity"
    )
    semfallback_eps: float = 0.10
    pdiff_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    pdiff_device: str = "cuda:3"
    telegraphic_eps: float = 0.12
    telegraphic_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    telegraphic_device: str = "cuda:3"
    ratediff_eps: float = 0.05
    superpose_model: str = "data/superpose_subspace/final"
    superpose_device: str = "cuda:3"
    # None -> read from the checkpoint's superpose_config.json
    superpose_key_seed: int | None = None
    superpose_key_mode: str | None = None
    superpose_max_slots: int = 8

    def compressor_params(self) -> dict[str, Any]:
        """Parameters for the selected compressor."""
        if self.compressor == "window":
            return {"window_rounds": self.window_rounds}
        if self.compressor == "novelty":
            return {"threshold": self.novelty_threshold, "stateful": self.novelty_stateful}
        if self.compressor == "llmlingua2":
            return {"rate": self.lingua_rate}
        if self.compressor == "learned":
            return {
                "rate": self.selector_rate,
                "dedup": self.selector_dedup,
                "model_path": self.selector_model,
            }
        if self.compressor == "rewriter":
            return {
                "rate": self.rewriter_rate,
                "model_path": self.rewriter_model,
            }
        if self.compressor == "backref":
            return {
                "threshold": self.backref_threshold,
                "drop_floor": self.backref_drop_floor,
                "model_path": self.selector_model,
            }
        if self.compressor == "codebook":
            return {"min_count": self.codebook_min_count}
        if self.compressor == "adaptive":
            return {
                "model_path": self.selector_model,
                "drop_floor": self.adaptive_drop_floor,
                "thr_hi": self.adaptive_thr_hi,
                "thr_lo": self.adaptive_thr_lo,
            }
        if self.compressor == "stack":
            return {"stack": self.stack}
        if self.compressor == "counterfactual":
            return {"tau": self.counterfactual_tau, "model_path": self.counterfactual_model}
        if self.compressor == "vib_sender":
            return {"rate": self.vib_rate, "model_path": self.vib_model}
        if self.compressor == "repmatch_selector":
            return {"tau": self.repmatch_tau, "model_path": self.repmatch_selector_model}
        if self.compressor == "saliency":
            return {"rate": self.saliency_rate, "repserver_url": self.repserver_url}
        if self.compressor == "repmatch_bestofk":
            return {"repserver_url": self.repserver_url}
        if self.compressor == "repmatch_rewriter":
            return {
                "rate": self.repmatch_rewriter_rate,
                "model_path": self.repmatch_rewriter_model,
            }
        if self.compressor == "tokenfilter":
            return {
                "tau": self.tokenfilter_tau,
                "model_path": self.tokenfilter_model,
                "device": self.tokenfilter_device,
            }
        if self.compressor == "autoencoder":
            return {
                "model_path": self.autoencoder_model,
                "num_latents": self.autoencoder_num_latents,
                "device": self.autoencoder_device,
            }
        if self.compressor == "mwnot_autoencoder":
            return {
                "model_path": self.mwnot_autoencoder_model,
                "num_latents": self.mwnot_autoencoder_num_latents,
                "device": self.mwnot_autoencoder_device,
            }
        if self.compressor == "grammar":
            return {
                "codebook_path": self.grammar_codebook,
            }
        if self.compressor == "certspan":
            return {"eps": self.certspan_eps, "repserver_url": self.repserver_url}
        if self.compressor == "semfallback":
            return {
                "ladder": self.semfallback_ladder,
                "eps": self.semfallback_eps,
                "repserver_url": self.repserver_url,
            }
        if self.compressor == "pdiff":
            return {"model_path": self.pdiff_model, "device": self.pdiff_device}
        if self.compressor == "telegraphic":
            return {
                "eps": self.telegraphic_eps,
                "model_path": self.telegraphic_model,
                "device": self.telegraphic_device,
                "repserver_url": self.repserver_url,
            }
        if self.compressor == "ratediff":
            return {
                "eps": self.ratediff_eps,
                "model_path": self.pdiff_model,
                "device": self.pdiff_device,
                "repserver_url": self.repserver_url,
            }
        if self.compressor == "superpose":
            return {
                "model_path": self.superpose_model,
                "device": self.superpose_device,
                "key_seed": self.superpose_key_seed,
                "key_mode": self.superpose_key_mode,
                "max_slots": self.superpose_max_slots,
            }
        return {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "compressor": self.compressor,
            "message_style": self.message_style,
            **self.compressor_params(),
        }


@dataclass
class AdvancedConfig:
    """Advanced configuration options."""
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 0
    log_level: str = "INFO"
    save_intermediate: bool = True
    parallel_tasks: int = 1


@dataclass
class Config:
    """
    Main configuration class for HiddenBench.

    Loads configuration from YAML files and environment variables.
    """
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """
        Load configuration from file and/or environment.

        Priority (highest to lowest):
        1. Environment variables
        2. Config file
        3. Default values

        Args:
            path: Path to config file. If None, looks for config.yaml
                  in current directory.

        Returns:
            Loaded configuration.
        """
        config = cls()

        # Try to load from file
        if path:
            config._load_from_file(Path(path))
        else:
            # Look for config in common locations
            for check_path in ["config.yaml", "config.yml", "config.json"]:
                if Path(check_path).exists():
                    config._load_from_file(Path(check_path))
                    break

        # Override with environment variables
        config._load_from_env()

        return config

    def _load_from_file(self, path: Path) -> None:
        """Load configuration from a file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            if path.suffix == ".json":
                import json
                data = json.load(f)
            else:
                data = yaml.safe_load(f)

        if not data:
            return

        # Load provider configs
        if "providers" in data:
            for name, pconfig in data["providers"].items():
                if pconfig:
                    # Known fields that shouldn't go into extra
                    known_fields = {
                        "api_key", "default_model", "base_url", "timeout",
                        "est_base_input_tokens", "est_output_tokens", "est_context_growth"
                    }
                    self.providers[name] = ProviderConfig(
                        api_key=pconfig.get("api_key", ""),
                        default_model=pconfig.get("default_model", ""),
                        base_url=pconfig.get("base_url"),
                        timeout=pconfig.get("timeout"),
                        est_base_input_tokens=pconfig.get("est_base_input_tokens", 800),
                        est_output_tokens=pconfig.get("est_output_tokens", 250),
                        est_context_growth=pconfig.get("est_context_growth", 300),
                        extra={
                            k: v for k, v in pconfig.items()
                            if k not in known_fields
                        },
                    )

        # Load benchmark config
        if "benchmark" in data:
            bconfig = data["benchmark"]
            self.benchmark = BenchmarkConfig(
                provider=bconfig.get("provider", self.benchmark.provider),
                model=bconfig.get("model"),
                num_agents=bconfig.get("num_agents", self.benchmark.num_agents),
                num_rounds=bconfig.get("num_rounds", self.benchmark.num_rounds),
                temperature=bconfig.get("temperature", self.benchmark.temperature),
                max_tokens=bconfig.get("max_tokens", self.benchmark.max_tokens),
                data_dir=bconfig.get("data_dir", self.benchmark.data_dir),
                tasks_dir=bconfig.get("tasks_dir", self.benchmark.tasks_dir),
                reports_dir=bconfig.get("reports_dir", self.benchmark.reports_dir),
                run_full_profile=bconfig.get("run_full_profile", self.benchmark.run_full_profile),
                seed=bconfig.get("seed"),
                use_official_data=bconfig.get("use_official_data", self.benchmark.use_official_data),
                use_custom_tasks=bconfig.get("use_custom_tasks", self.benchmark.use_custom_tasks),
            )

        # Load channel config
        if "channel" in data:
            cconfig = data["channel"]
            self.channel = ChannelConfig(
                protocol=cconfig.get("protocol", self.channel.protocol),
                compressor=cconfig.get("compressor", self.channel.compressor),
                message_style=cconfig.get("message_style", self.channel.message_style),
                window_rounds=cconfig.get("window_rounds", self.channel.window_rounds),
                novelty_threshold=cconfig.get("novelty_threshold", self.channel.novelty_threshold),
                novelty_stateful=cconfig.get("novelty_stateful", self.channel.novelty_stateful),
                lingua_rate=cconfig.get("lingua_rate", self.channel.lingua_rate),
                selector_rate=cconfig.get("selector_rate", self.channel.selector_rate),
                selector_dedup=cconfig.get("selector_dedup", self.channel.selector_dedup),
                selector_model=cconfig.get("selector_model", self.channel.selector_model),
                rewriter_rate=cconfig.get("rewriter_rate", self.channel.rewriter_rate),
                rewriter_model=cconfig.get("rewriter_model", self.channel.rewriter_model),
                backref_threshold=cconfig.get("backref_threshold", self.channel.backref_threshold),
                backref_drop_floor=cconfig.get("backref_drop_floor", self.channel.backref_drop_floor),
                codebook_min_count=cconfig.get("codebook_min_count", self.channel.codebook_min_count),
                adaptive_drop_floor=cconfig.get("adaptive_drop_floor", self.channel.adaptive_drop_floor),
                adaptive_thr_hi=cconfig.get("adaptive_thr_hi", self.channel.adaptive_thr_hi),
                adaptive_thr_lo=cconfig.get("adaptive_thr_lo", self.channel.adaptive_thr_lo),
                stack=cconfig.get("stack", self.channel.stack),
                counterfactual_tau=cconfig.get("counterfactual_tau", self.channel.counterfactual_tau),
                counterfactual_model=cconfig.get("counterfactual_model", self.channel.counterfactual_model),
                vib_rate=cconfig.get("vib_rate", self.channel.vib_rate),
                vib_model=cconfig.get("vib_model", self.channel.vib_model),
                repmatch_tau=cconfig.get("repmatch_tau", self.channel.repmatch_tau),
                repmatch_selector_model=cconfig.get(
                    "repmatch_selector_model", self.channel.repmatch_selector_model
                ),
                saliency_rate=cconfig.get("saliency_rate", self.channel.saliency_rate),
                repserver_url=cconfig.get("repserver_url", self.channel.repserver_url),
                repmatch_rewriter_rate=cconfig.get(
                    "repmatch_rewriter_rate", self.channel.repmatch_rewriter_rate
                ),
                repmatch_rewriter_model=cconfig.get(
                    "repmatch_rewriter_model", self.channel.repmatch_rewriter_model
                ),
                tokenfilter_tau=cconfig.get(
                    "tokenfilter_tau", self.channel.tokenfilter_tau
                ),
                tokenfilter_model=cconfig.get(
                    "tokenfilter_model", self.channel.tokenfilter_model
                ),
                tokenfilter_device=cconfig.get(
                    "tokenfilter_device", self.channel.tokenfilter_device
                ),
                autoencoder_model=cconfig.get(
                    "autoencoder_model", self.channel.autoencoder_model
                ),
                autoencoder_num_latents=cconfig.get(
                    "autoencoder_num_latents", self.channel.autoencoder_num_latents
                ),
                autoencoder_device=cconfig.get(
                    "autoencoder_device", self.channel.autoencoder_device
                ),
                mwnot_autoencoder_model=cconfig.get(
                    "mwnot_autoencoder_model", self.channel.mwnot_autoencoder_model
                ),
                mwnot_autoencoder_num_latents=cconfig.get(
                    "mwnot_autoencoder_num_latents", self.channel.mwnot_autoencoder_num_latents
                ),
                mwnot_autoencoder_device=cconfig.get(
                    "mwnot_autoencoder_device", self.channel.mwnot_autoencoder_device
                ),
                grammar_codebook=cconfig.get(
                    "grammar_codebook", self.channel.grammar_codebook
                ),
                certspan_eps=cconfig.get("certspan_eps", self.channel.certspan_eps),
                semfallback_ladder=cconfig.get(
                    "semfallback_ladder", self.channel.semfallback_ladder
                ),
                semfallback_eps=cconfig.get("semfallback_eps", self.channel.semfallback_eps),
                pdiff_model=cconfig.get("pdiff_model", self.channel.pdiff_model),
                pdiff_device=cconfig.get("pdiff_device", self.channel.pdiff_device),
                telegraphic_eps=cconfig.get("telegraphic_eps", self.channel.telegraphic_eps),
                telegraphic_model=cconfig.get(
                    "telegraphic_model", self.channel.telegraphic_model
                ),
                telegraphic_device=cconfig.get(
                    "telegraphic_device", self.channel.telegraphic_device
                ),
                ratediff_eps=cconfig.get("ratediff_eps", self.channel.ratediff_eps),
            )

        # Load advanced config
        if "advanced" in data:
            aconfig = data["advanced"]
            self.advanced = AdvancedConfig(
                max_retries=aconfig.get("max_retries", self.advanced.max_retries),
                retry_delay=aconfig.get("retry_delay", self.advanced.retry_delay),
                rate_limit=aconfig.get("rate_limit", self.advanced.rate_limit),
                log_level=aconfig.get("log_level", self.advanced.log_level),
                save_intermediate=aconfig.get("save_intermediate", self.advanced.save_intermediate),
                parallel_tasks=aconfig.get("parallel_tasks", self.advanced.parallel_tasks),
            )

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Provider API keys from environment
        env_mappings = {
            "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
            "OPENAI_API_KEY": ("openai", "api_key"),
            "GROK_API_KEY": ("grok", "api_key"),
            "XAI_API_KEY": ("grok", "api_key"),
        }

        for env_var, (provider, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                if provider not in self.providers:
                    self.providers[provider] = ProviderConfig()
                setattr(self.providers[provider], key, value)

        # Benchmark settings from environment
        if os.environ.get("HIDDENBENCH_PROVIDER"):
            self.benchmark.provider = os.environ["HIDDENBENCH_PROVIDER"]
        if os.environ.get("HIDDENBENCH_MODEL"):
            self.benchmark.model = os.environ["HIDDENBENCH_MODEL"]

    def get_provider_config(self, provider_name: str) -> dict[str, Any]:
        """
        Get configuration dictionary for a provider.

        Args:
            provider_name: Name of the provider.

        Returns:
            Configuration dictionary.

        Raises:
            ValueError: If provider is not configured.
        """
        if provider_name not in self.providers:
            raise ValueError(
                f"Provider '{provider_name}' not configured. "
                f"Please add configuration for this provider in config.yaml"
            )

        return self.providers[provider_name].to_dict()

    def get_token_estimation_params(self, provider_name: str | None = None) -> dict[str, int]:
        """
        Get token estimation parameters for a provider.

        Args:
            provider_name: Name of the provider. If None, uses benchmark provider.

        Returns:
            Dictionary with est_base_input_tokens, est_output_tokens, est_context_growth.
        """
        provider_name = provider_name or self.benchmark.provider

        # Default values (conservative estimates)
        defaults = {
            "est_base_input_tokens": 800,
            "est_output_tokens": 250,
            "est_context_growth": 300,
        }

        if provider_name in self.providers:
            pconfig = self.providers[provider_name]
            return {
                "est_base_input_tokens": pconfig.est_base_input_tokens,
                "est_output_tokens": pconfig.est_output_tokens,
                "est_context_growth": pconfig.est_context_growth,
            }

        return defaults

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if valid.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Check that the configured provider exists
        provider = self.benchmark.provider
        if provider not in self.providers:
            raise ValueError(
                f"Benchmark provider '{provider}' not configured. "
                f"Add its configuration under 'providers:' in config.yaml"
            )

        # Check that provider has API key (except for local)
        if provider != "local":
            pconfig = self.providers[provider]
            if not pconfig.api_key or pconfig.api_key.startswith("your-"):
                raise ValueError(
                    f"Provider '{provider}' requires a valid API key. "
                    f"Set it in config.yaml or via environment variable."
                )

        return True
