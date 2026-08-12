# MWNOT autoencoder

The MWNOT variant replaces fixed-position latent sampling with a learned
multiwavelet sequence generator. Every input position can contribute to the
fixed-size generator vectors that condition reconstruction.

`run_full.sh` creates text data, trains the joint language-model/generator
container, validates the operator and held-out reconstruction, then compares
the checkpoint with identity on HiddenBench:

```bash
BENCH_CONFIG=experiments/hiddenbench/configs/config.example.yaml \
DEVICE=cuda:0 STEPS=3000 bash experiments/mwnot/run_full.sh
```

Stages are `data`, `train`, `validate`, and `benchmark`. Watch both base-model
and generator gradients, finite training/dev loss, and held-out generations.
The checkpoint's `generator.pt` stores the operator state and constructor
configuration; a checkpoint is incomplete without it.

MWNOT adds modeling capacity, not packet capacity. Compare it with the sampled
autoencoder at the same latent count, data split, base model, and benchmark
tasks.
