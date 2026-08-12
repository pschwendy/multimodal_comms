# Training

This directory owns all dataset construction and parameter optimization. It is
independent of benchmark runners: trainers consume explicit files or injected
services and produce checkpoints; experiments decide which benchmark later
uses those checkpoints.

## Layout

| Directory | Responsibility |
|---|---|
| `data/` | harvest and transform JSON/JSONL corpora |
| `programs/` | autoencoder, packing, selector, policy, GKD, and keyring training |
| `services/` | local representation service used by semantic objectives |
| `validation/` | reconstruction and checkpoint-level qualitative checks |
| `benchmark_adaptation/` | optional COMMA/iAgents training-data adapters |

Most text corpora use one JSON object per line with a `text` field. Selector
and policy datasets have richer schemas documented in their builder's module
docstring and `--help`. Generated data and model state should stay under
ignored `outputs/` paths.

## Autoencoders

Create a disjoint train/dev corpus, train, then inspect held-out
reconstructions:

```bash
python -m training.data.harvest_fineweb_data \
  --train-examples 60000 --dev-examples 2000 \
  --out-dir outputs/data/fineweb_ae

python -m training.programs.pretrain_autoencoder \
  --train-data outputs/data/fineweb_ae/train.jsonl \
  --dev-data outputs/data/fineweb_ae/dev.jsonl \
  --out-dir outputs/models/autoencoder --device cuda:0

python -m training.validation.eval_autoencoder_qualitative \
  --checkpoint outputs/models/autoencoder/final \
  --dev-data outputs/data/fineweb_ae/dev.jsonl --device cuda:0
```

`pretrain_mwnot_autoencoder` replaces sampled positions with the MWNOT
sequence generator. `pretrain_packed` adds a nested bottleneck for Block and
Frame packing. `pretrain_superpose` teaches the decoder to tolerate keyed
superposition crosstalk. `pretrain_image_packed` trains the image-side code.

## Selectors and policies

The sentence selector path is harvest -> build features ->
`train_selector`. Representation matching uses
`harvest_repmatch_data` -> `build_repmatch_dataset` ->
`train_repmatch_selector`, or a policy dataset -> `train_repmatch_grpo`.
Counterfactual, VIB, rewriter, and token-filter objectives have named
harvest/train pairs in the same directories.

Some harvesters consume completed discussion reports because their labels are
defined by a receiver's behavior. That does not make them benchmark runners:
they read static reports and emit ordinary training data. Endpoint, input, and
output paths should be supplied for the experiment being run.

## GKD

GKD consists of three distinct training programs:

```text
gkd_rollout -> gkd_score -> gkd_train
```

The first samples from a student checkpoint, the second obtains teacher
distributions, and the third performs the KL update. The full multi-GPU
trajectory is documented in `experiments/gkd/README.md`.

## Deciding whether training worked

During training, require finite loss, a decreasing held-out reconstruction or
objective loss, and checkpoints that can reload. After training, use the
matching program in `training/validation/`, then run the end-to-end experiment
against an identity baseline. A lower training loss alone is not evidence of
better communication.

Before a long run, execute the objective-level progress tests:

```bash
python -m experiments.validation.training_dynamics
```

The catalog in `experiments/validation/training_catalog.yaml` maps every
training entry point to a tiny learnable probe. These probes test optimization
wiring and natural progress, not large-model quality.
