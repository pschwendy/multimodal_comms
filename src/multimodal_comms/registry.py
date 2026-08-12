from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

import numpy as np

from .methods.autoencoders import AutoencoderConfig, AutoencoderMethod, MWNOTAutoencoderMethod
from .methods.multimodal import ImageZlibCodec, MixedPacketCodec
from .methods.packing import BlockPacker, FramePacker, RotorPacker
from .methods.predictive import (
    PredictiveDiffConfig,
    PredictiveDiffMethod,
    RateDiffConfig,
    RateDiffMethod,
    TelegraphicConfig,
    TelegraphicMethod,
)
from .methods.sensing import (
    CompressedSensingCodec,
    CURCodec,
    PCACodec,
    SVDCodec,
    make_sensing_matrix,
)
from .methods.superposition import SuperpositionPacker
from .methods.text import (
    BackrefConfig,
    BackrefMethod,
    CodebookConfig,
    CodebookMethod,
    ExtractiveConfig,
    ExtractiveMethod,
    GzipMethod,
    IdentityMethod,
    NoveltyConfig,
    NoveltyMethod,
    RewriterMethod,
    WindowConfig,
    WindowMethod,
    TextCompressorConfig,
    TextCompressorMethod,
    build_text_method,
)

MethodKind = Literal["communication", "codec", "packer"]


@dataclass(frozen=True, slots=True)
class NoConfig:
    pass


@dataclass(frozen=True, slots=True)
class PackerConfig:
    packet_dim: int = 16
    code_dim: int = 4
    seed: int = 1234
    nonce: int | None = None


@dataclass(frozen=True, slots=True)
class SuperpositionConfig:
    dim: int = 8
    seed: int = 1234
    nonce: int | None = None
    row_keys: bool = True


@dataclass(frozen=True, slots=True)
class SensingConfig:
    dimension: int = 8
    measurements: int = 8
    sparsity: int | None = None
    recovery: Literal["omp", "ridge"] = "omp"
    ridge: float = 1e-8
    seed: int = 0


@dataclass(frozen=True, slots=True)
class RankConfig:
    rank: int = 2


@dataclass(frozen=True, slots=True)
class MethodSpec:
    id: str
    implementation: type
    config_type: type
    dependencies: tuple[str, ...]
    representation: tuple[str, ...]
    kind: MethodKind
    description: str
    builder: Callable[[Any, Mapping[str, Any]], Any]

    def missing_dependencies(self) -> tuple[str, ...]:
        return tuple(name for name in self.dependencies if importlib.util.find_spec(name) is None)


def _config(config_type: type, values: Mapping[str, Any] | Any | None):
    if values is None:
        return config_type()
    if isinstance(values, config_type):
        return values
    if not isinstance(values, Mapping):
        raise TypeError(f"configuration must be {config_type.__name__} or a mapping")
    allowed = {field.name for field in fields(config_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown configuration keys for {config_type.__name__}: {sorted(unknown)}"
        )
    return config_type(**values)


def _standard(cls):
    return lambda config, injected: cls(config, **injected)


def _simple(cls):
    return lambda config, injected: cls(**injected)


def _packer(cls):
    def build(config: PackerConfig, injected):
        values = dataclasses.asdict(config) | dict(injected)
        if cls is BlockPacker:
            values = {key: values[key] for key in ("packet_dim", "code_dim")}
        elif cls is FramePacker:
            values.pop("layout_seed", None)
        elif cls is RotorPacker:
            values["layout_seed"] = values.pop("seed")
        return cls(**values)

    return build


def _sensing(config: SensingConfig, injected):
    phi = injected.get(
        "phi", make_sensing_matrix(config.measurements, config.dimension, config.seed)
    )
    dictionary = injected.get("dictionary", np.eye(config.dimension))
    return CompressedSensingCodec(
        phi,
        dictionary,
        sparsity=config.sparsity,
        recovery=config.recovery,
        ridge=config.ridge,
    )


def _spec(identifier, cls, config, deps, representations, kind, description, builder):
    return MethodSpec(
        identifier, cls, config, tuple(deps), tuple(representations), kind, description, builder
    )


_EXTRACTIVE_IDS = {
    "llmlingua2": "LLMLingua-style token pruning",
    "learned": "learned sentence selector",
    "adaptive": "adaptive extractive selection",
    "counterfactual": "counterfactual importance selection",
    "vib_sender": "variational information-bottleneck sender",
    "repmatch_selector": "representation-match selector",
    "saliency": "saliency pruning",
    "repmatch_bestofk": "representation-match best-of-k",
    "tokenfilter": "token-filter selection",
    "grammar": "shared grammar/codebook compression",
    "certspan": "certified span deletion",
    "semfallback": "semantic fallback ladder",
}


METHODS: dict[str, MethodSpec] = {
    "identity": _spec(
        "identity",
        IdentityMethod,
        NoConfig,
        (),
        ("text", "bytes"),
        "communication",
        "lossless canonical transport",
        _simple(IdentityMethod),
    ),
    "window": _spec(
        "window",
        WindowMethod,
        WindowConfig,
        (),
        ("text", "bytes"),
        "communication",
        "recent-round selection",
        _standard(WindowMethod),
    ),
    "novelty": _spec(
        "novelty",
        NoveltyMethod,
        NoveltyConfig,
        (),
        ("text",),
        "communication",
        "sentence novelty selection",
        _standard(NoveltyMethod),
    ),
    "backref": _spec(
        "backref",
        BackrefMethod,
        BackrefConfig,
        (),
        ("text", "bytes"),
        "communication",
        "lossless dictionary backreferences",
        _standard(BackrefMethod),
    ),
    "gzip64": _spec(
        "gzip64",
        GzipMethod,
        NoConfig,
        (),
        ("text", "bytes"),
        "communication",
        "gzip and base64 transport",
        _simple(GzipMethod),
    ),
    "codebook": _spec(
        "codebook",
        CodebookMethod,
        CodebookConfig,
        (),
        ("text",),
        "communication",
        "online or corpus codebook",
        _standard(CodebookMethod),
    ),
    "rewriter": _spec(
        "rewriter",
        RewriterMethod,
        NoConfig,
        (),
        ("text",),
        "communication",
        "injected message rewriter",
        _simple(RewriterMethod),
    ),
    "repmatch_rewriter": _spec(
        "repmatch_rewriter",
        RewriterMethod,
        NoConfig,
        (),
        ("text",),
        "communication",
        "representation-match rewriter",
        _simple(RewriterMethod),
    ),
    "stack": _spec(
        "stack",
        RewriterMethod,
        NoConfig,
        (),
        ("text",),
        "communication",
        "composable transformation stack",
        _simple(RewriterMethod),
    ),
    "pdiff": _spec(
        "pdiff",
        PredictiveDiffMethod,
        PredictiveDiffConfig,
        (),
        ("text", "bytes"),
        "communication",
        "lossless predictive corrections",
        _standard(PredictiveDiffMethod),
    ),
    "ratediff": _spec(
        "ratediff",
        RateDiffMethod,
        RateDiffConfig,
        (),
        ("text", "bytes"),
        "communication",
        "rate-controlled predictive corrections",
        _standard(RateDiffMethod),
    ),
    "telegraphic": _spec(
        "telegraphic",
        TelegraphicMethod,
        TelegraphicConfig,
        (),
        ("text",),
        "communication",
        "telegraphic stop-word elision",
        _standard(TelegraphicMethod),
    ),
    "autoencoder": _spec(
        "autoencoder",
        AutoencoderMethod,
        AutoencoderConfig,
        (),
        ("text", "latent"),
        "communication",
        "sampled-latent autoencoder",
        _standard(AutoencoderMethod),
    ),
    "mwnot_autoencoder": _spec(
        "mwnot_autoencoder",
        MWNOTAutoencoderMethod,
        AutoencoderConfig,
        (),
        ("text", "latent"),
        "communication",
        "MWNOT sequence autoencoder",
        _standard(MWNOTAutoencoderMethod),
    ),
    "block": _spec(
        "block",
        BlockPacker,
        PackerConfig,
        (),
        ("numeric",),
        "packer",
        "disjoint exact block packing",
        _packer(BlockPacker),
    ),
    "frame": _spec(
        "frame",
        FramePacker,
        PackerConfig,
        (),
        ("numeric",),
        "packer",
        "overloadable random-frame packing",
        _packer(FramePacker),
    ),
    "rotor": _spec(
        "rotor",
        RotorPacker,
        PackerConfig,
        (),
        ("numeric",),
        "packer",
        "dense exact rotated block packing",
        _packer(RotorPacker),
    ),
    "superpose": _spec(
        "superpose",
        SuperpositionPacker,
        SuperpositionConfig,
        (),
        ("numeric", "latent"),
        "packer",
        "overlapping keyed-code superposition",
        lambda c, i: SuperpositionPacker(**(dataclasses.asdict(c) | dict(i))),
    ),
    "compressed_sensing": _spec(
        "compressed_sensing",
        CompressedSensingCodec,
        SensingConfig,
        (),
        ("numeric",),
        "codec",
        "shared-dictionary compressed sensing",
        _sensing,
    ),
    "svd": _spec(
        "svd",
        SVDCodec,
        RankConfig,
        (),
        ("numeric",),
        "codec",
        "truncated SVD",
        lambda c, i: SVDCodec(c.rank, **i),
    ),
    "pca": _spec(
        "pca",
        PCACodec,
        RankConfig,
        (),
        ("numeric",),
        "codec",
        "principal components",
        lambda c, i: PCACodec(c.rank, **i),
    ),
    "cur": _spec(
        "cur",
        CURCodec,
        RankConfig,
        (),
        ("numeric",),
        "codec",
        "CUR decomposition",
        lambda c, i: CURCodec(c.rank, **i),
    ),
    "image_zlib": _spec(
        "image_zlib",
        ImageZlibCodec,
        NoConfig,
        (),
        ("image-bytes",),
        "codec",
        "lossless encoded-image compression",
        _simple(ImageZlibCodec),
    ),
    "mixed_packet": _spec(
        "mixed_packet",
        MixedPacketCodec,
        NoConfig,
        (),
        ("mixed",),
        "codec",
        "typed mixed-modality packet",
        _simple(MixedPacketCodec),
    ),
}

for identifier, description in _EXTRACTIVE_IDS.items():
    METHODS[identifier] = _spec(
        identifier,
        ExtractiveMethod,
        ExtractiveConfig,
        (),
        ("text",),
        "communication",
        description,
        _standard(ExtractiveMethod),
    )

# The entries above retain the small dependency-free reference methods used by
# focused unit tests.  Public experiment IDs, however, resolve to the complete
# text implementations through one canonical adapter.  This is
# deliberately data-driven: benchmark runners never contain method-specific
# construction branches.
_REGISTERED_COMMUNICATION = {
    "identity": ("lossless canonical transport", ()),
    "window": ("recent-round selection", ()),
    "novelty": ("sentence novelty selection", ("sentence_transformers",)),
    "llmlingua2": ("LLMLingua-style token pruning", ("llmlingua",)),
    "learned": ("learned sentence selector", ("joblib", "sentence_transformers")),
    "counterfactual": ("counterfactual importance selection", ("joblib", "sentence_transformers")),
    "rewriter": ("GRPO-trained abstractive rewriter", ("torch", "transformers")),
    "backref": ("lossless dictionary backreferences", ()),
    "codebook": ("online or corpus codebook", ()),
    "adaptive": ("adaptive extractive selection", ("joblib", "sentence_transformers")),
    "gzip64": ("gzip and base64 transport", ()),
    "stack": ("composable transformation stack", ()),
    "vib_sender": ("variational information-bottleneck sender", ("torch", "transformers")),
    "repmatch_selector": ("representation-match selector", ("joblib", "sentence_transformers")),
    "saliency": ("gradient-saliency token pruning", ("requests",)),
    "repmatch_bestofk": ("representation-match best-of-k", ("requests",)),
    "repmatch_rewriter": ("representation-match GRPO rewriter", ("torch", "transformers")),
    "tokenfilter": ("policy-gradient token filter", ("torch", "transformers")),
    "autoencoder": ("sampled-latent sequence autoencoder", ("torch", "transformers")),
    "mwnot_autoencoder": ("MWNOT sequence autoencoder", ("torch", "transformers")),
    "grammar": ("shared grammar/codebook compression", ()),
    "certspan": ("certified span deletion", ("requests",)),
    "semfallback": ("semantic fallback ladder", ("requests",)),
    "pdiff": ("lossless model-predictive corrections", ("torch", "transformers")),
    "ratediff": ("rate-controlled predictive corrections", ("torch", "transformers", "requests")),
    "telegraphic": ("certified telegraphic generation", ("torch", "transformers", "requests")),
    "superpose": ("keyed latent-code superposition", ("torch", "transformers")),
}

for identifier, (description, dependencies) in _REGISTERED_COMMUNICATION.items():
    representations = ("text", "latent") if identifier in {
        "autoencoder", "mwnot_autoencoder", "superpose"
    } else ("text",)
    METHODS[identifier] = _spec(
        identifier,
        TextCompressorMethod,
        TextCompressorConfig,
        dependencies,
        representations,
        "communication",
        description,
        lambda config, injected, method_id=identifier: build_text_method(
            method_id, config, injected
        ),
    )


def list_methods(*, kind: MethodKind | None = None) -> tuple[MethodSpec, ...]:
    return tuple(
        METHODS[key] for key in sorted(METHODS) if kind is None or METHODS[key].kind == kind
    )


def get_method_spec(identifier: str) -> MethodSpec:
    try:
        return METHODS[identifier]
    except KeyError as error:
        raise KeyError(
            f"unknown method {identifier!r}; available: {', '.join(sorted(METHODS))}"
        ) from error


def create_method(identifier: str, config: Mapping[str, Any] | Any | None = None, **injected):
    spec = get_method_spec(identifier)
    typed = _config(spec.config_type, config)
    return spec.builder(typed, injected)
