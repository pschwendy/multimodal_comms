# multimodal_comms

`multimodal_comms` is a research monorepo for communication under bandwidth,
latency, and privacy constraints. It contains reusable text and multimodal
communication methods, the code that trains learned methods, complete
experiment trajectories, and four information-asymmetry benchmarks.

The current research emphasis is learned compress–decompress communication:
encode messages into compact latent packets, reconstruct them for receivers,
and measure whether the recovered information still supports collaborative
task completion. See the [project summary](PROJECT_SUMMARY.md) for the goal and
the relationship between autoencoders, packing, superposition, and benchmark
evaluation.

You do not need to install this repository as a Python package. Run commands
from its root with the repository and `src/` on `PYTHONPATH`.

## Environment

The base Conda environment runs the CPU tests and dependency-light examples:

```bash
conda env create -f environment.yml
conda activate multimodal-comms
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m multimodal_comms.cli.main methods
python -m multimodal_comms.cli.main roundtrip identity "meeting starts at noon"
python -m experiments.smoke --profile core
```

Use the full environment for neural methods and applications:

```bash
conda env create -f environment-full.yml
conda activate multimodal-comms-full
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.smoke --profile full
```

The full environment is CPU-compatible. Before multi-GPU training, install the
PyTorch build matching the machine's CUDA driver. GKD teacher scoring also
requires `vllm`; local GGUF inference requires a host-appropriate build of
`llama-cpp-python`; Windows COMMA capture requires `pywin32`.

Keep credentials in environment variables, never config files. Depending on
the provider, runs may use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY`, `AZURE_OPENAI_API_KEY`, or `GOOGLE_API_KEY`.

## Repository map

```text
src/multimodal_comms/core/       messages, contexts, transmissions, interfaces
src/multimodal_comms/methods/    reusable communication algorithms
src/multimodal_comms/benchmarks/ adapters, task execution, and native grading
src/multimodal_comms/apps/       COMMA, Collab-Overcooked, and iAgents apps
training/data/                   dataset construction and harvesting
training/programs/               model training entry points
training/services/               representation and teacher services
training/validation/             checkpoint-level validation
experiments/                     complete data-to-result trajectories
docs/results/                    concise findings and caveats
docs/proofs/                     derivations and scientific corrections
tests/                           contracts, invariants, and smoke tests
```

The architectural rule is simple: methods do not know about benchmarks, and
training does not live inside a benchmark. Benchmark-specific adaptation data
is under `training/benchmark_adaptation/`, not in the benchmark runner.
Generated datasets, checkpoints, logs, and reports belong in ignored
`outputs/`, `artifacts/`, or `checkpoints/` directories.

Each method has a short README beside its source. Start at
[`src/multimodal_comms/methods`](src/multimodal_comms/methods), then use the
experiment README for the complete training and evaluation path.

## Running a complete experiment

Every full experiment has `experiments/<name>/README.md` and
`experiments/<name>/run_full.sh`. The README explains inputs, stages, metrics,
and failure modes. The shell file is the executable trajectory and keeps the
commands in their actual order:

1. construct or harvest training data;
2. train the method and save checkpoints;
3. validate reconstruction, optimization, or method invariants;
4. insert the checkpoint into an information-asymmetry benchmark;
5. aggregate native task quality together with communication traffic.

Use `STAGE=<name>` to run one stage when a launcher supports staged execution.
Run `bash -n experiments/<name>/run_full.sh` to inspect syntax without starting
work. Real runs require the models, datasets, services, and credentials named
in that experiment's README.

| Experiment | What it establishes |
|---|---|
| [Autoencoders](experiments/autoencoders/README.md) | sampled-latent training, reconstruction validation, HiddenBench use |
| [CryptAE](experiments/crypt_ae/README.md) | keyed superposition training, fidelity, leakage, load, benchmark use |
| [MWNOT](experiments/mwnot/README.md) | sequence-generator autoencoder training and reconstruction |
| [Selectors](experiments/selectors/README.md) | data harvesting, selector/rewriter training, benchmark sweeps |
| [GKD](experiments/gkd/README.md) | rollout, teacher scoring, KL training, probe evaluation |
| [Packing](experiments/packing/README.md) | packed latent training and load/crosstalk evaluation |
| [Compressed sensing](experiments/compressed_sensing/README.md) | image positive controls and embedding negative controls |
| [HiddenBench](experiments/hiddenbench/README.md) | asynchronous hidden-information discussion |
| [COMMA](experiments/comma/README.md) | multimodal collaborative puzzles |
| [Collab-Overcooked](experiments/collab_overcooked/README.md) | embodied coordination and communication |
| [iAgents](experiments/iagents/README.md) | offline information-asymmetry questions and independent regrading |

For a first real run, copy an experiment launcher, override its paths through
environment variables, and start with a small task count. Do not interpret a
lower byte count alone as improvement. Always compare communication cost with
the benchmark's native success metric and with `identity` under the same model,
seed, task subset, and discussion policy.

## Methods and benchmarks

The method registry is the single construction path used by benchmark
adapters. `full_history` and `delta` are channel-view policies, not compression
methods.

| Family | Stable method IDs |
|---|---|
| Text | `identity`, `window`, `novelty`, `llmlingua2`, `learned`, `rewriter`, `backref`, `codebook`, `adaptive`, `gzip64`, `stack`, `counterfactual`, `vib_sender`, `repmatch_selector`, `saliency`, `repmatch_bestofk`, `repmatch_rewriter`, `tokenfilter`, `grammar`, `certspan`, `semfallback` |
| Predictive | `pdiff`, `ratediff`, `telegraphic` |
| Autoencoder | `autoencoder`, `mwnot_autoencoder` |
| Superposition | `superpose` |
| Packing | `block`, `frame`, `rotor` |
| Sensing | `compressed_sensing`, `svd`, `pca`, `cur` |
| Multimodal | `image_zlib`, `mixed_packet` |

| Benchmark | Native outcomes |
|---|---|
| HiddenBench | pre/post accuracy, information gain, consensus, traffic |
| COMMA | puzzle completion and Telehealth partial credit |
| Collab-Overcooked | success, TES/F1, similarity, redundancy, collaboration |
| iAgents | answer grading, independent regrade, traffic |

## Verification

Run the repository acceptance checks before a long experiment:

```bash
python -m experiments.smoke --profile core
pytest -q
python scripts/check_repository.py
```

For learned and sensing methods, also run:

```bash
python -m experiments.validation.training_dynamics
python -m experiments.validation.sensing_behavior --require-natural-image
```

The training validation uses tiny learnable problems to catch detached
gradients, non-finite objectives, and loss trajectories that do not improve.
The sensing validation is modality-aware: sparse image blocks must reconstruct
well, while dense generic embedding matrices are expected to reconstruct poorly
under a sparse compressed-sensing prior. For embeddings, compare against
ridge, PCA/SVD, and held-out task performance instead of assuming sparsity.

The smoke harness compiles the source, training programs, experiment programs,
and shell trajectories; validates configs; exercises dependency-light kernels;
and reports optional checks it could not run. It does not claim to reproduce a
large-model result without the corresponding model and data.

Scientific details and corrections, including row-key leakage, RotorPacker
capacity, predictive replay, and the autoencoder decode-embedding clone, are in
[`docs/proofs`](docs/proofs) and [`docs/results`](docs/results).
