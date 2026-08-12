# Generative knowledge distillation

GKD improves a latent-conditioned student in an on-policy loop: sample student
rollouts, obtain top-k teacher distributions, train with a token-level KL
objective, then probe the updated checkpoint on held-out latent examples.

```bash
bash experiments/gkd/run_full.sh 1 outputs/models/latent_reader/final
```

The launcher is an eight/nine-GPU trajectory. Override `PYTHON`, `TORCHRUN`,
and `CUDA_VISIBLE_DEVICES` for the machine. Each round writes rollouts, scored
teacher tensors, the updated model, logs, and probe results beneath
`outputs/gkd/roundN`.

Before scale-up, run `python -m experiments.validation.training_dynamics`; its
distribution-distillation probe verifies finite gradients and decreasing KL.
For a real round, verify rollout diversity, teacher score coverage, finite KL,
and held-out probe accuracy. Feed the resulting latent reader into the learned
communication experiment that produced its latent representation; GKD itself
is a training procedure, not a channel method ID.
