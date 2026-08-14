# HiddenBench

HiddenBench measures whether agents exchange private facts that are necessary
to solve a shared task. This directory contains only benchmark fixtures,
configuration, execution, sweeps, and result aggregation. Dataset creation and
model training are in `training/`; full learned-method trajectories are in the
other experiment directories.

See the [benchmark overview, paper, and preserved original README](../../src/multimodal_comms/benchmarks/hiddenbench/README.md).

## Run a benchmark condition

Copy `configs/config.example.yaml`, select a provider, and set credentials in
the environment. Then run an identity baseline:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
  --config experiments/hiddenbench/configs/config.example.yaml \
  --compressor identity --num-tasks 1 --rounds 1 --no-full-profile
```

For a learned compressor, put its checkpoint path in the copied config's
`channel` section and select its stable method ID. For example,
`autoencoder_model` configures `--compressor autoencoder`, and
`mwnot_autoencoder_model` configures `--compressor mwnot_autoencoder`.

`run_full.sh` runs identity and one selected method on the same task count and
then aggregates the reports. The `programs/run_*sweep.sh` files define the
larger condition grids.

## Interpretation

Report pre-discussion accuracy, post-discussion accuracy, information gain,
consensus, transmitted bytes/tokens, and method errors. Compare methods only
when the provider, model, task subset, rounds, seed, and channel-view policy are
the same. `full_history` and `delta` change what a receiver sees; they are not
compression algorithms.

The bundled fixture permits local smoke evaluation. Network-backed provider
runs remain explicit and require their endpoint and credential. HiddenBench
does not create training data or update model weights.
