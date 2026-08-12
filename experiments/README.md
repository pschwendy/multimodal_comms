# Experiments

This directory contains executable research trajectories. Every experiment
family has a short guide and a `run_full.sh` that shows the real stage order.
Run all commands from the repository root after setting:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Training implementation belongs in `training/`; reusable algorithms belong in
`src/multimodal_comms/methods/`; benchmark execution belongs in the benchmark
experiment. The launchers here connect those pieces without redefining them.

| Family | Full trajectory |
|---|---|
| Autoencoders | data -> train -> reconstruct -> HiddenBench |
| CryptAE | data -> base autoencoder -> keyed superposition -> security/load -> HiddenBench |
| MWNOT | data -> MWNOT autoencoder -> reconstruct -> HiddenBench |
| Selectors | harvest -> features/policy -> train -> asynchronous benchmark sweep |
| GKD | rollout -> teacher score -> KL update -> held-out probe |
| Packing | train packet model -> capacity/crosstalk/multimodal evaluation |
| Compressed sensing | validate prior -> image/embedding experiments -> compare baselines |
| HiddenBench | configure -> identity baseline -> method condition -> aggregate |
| COMMA | configure app -> run puzzles -> native grade -> aggregate |
| Collab-Overcooked | run episodes -> native evaluation -> aggregate |
| iAgents | offline questions -> method sweep -> independent regrade -> aggregate |

Use `python -m experiments.smoke --profile core` before a real run. Generated
datasets, models, logs, plots, and reports should be written under `outputs/`
or `artifacts/`; both are ignored by Git.
