# Behavioral validation

These checks answer two questions that import and one-step tests cannot:
whether each distinct training objective can make stable progress on a tiny
learnable problem, and whether sensing methods behave according to their
mathematical assumptions.

```bash
export PYTHONPATH="$PWD/src:$PWD"

# Base environment: NumPy/scikit-learn model families.
python -m experiments.validation.training_dynamics --base-only

# Full environment: all ten objective families with real small modules.
python -m experiments.validation.training_dynamics

# Positive image controls and a negative dense-embedding control.
python -m experiments.validation.sensing_behavior --require-natural-image
```

`training_catalog.yaml` maps every training entry point to its behavioral
probe. The full run exercises linear fitting, selector classification and
regression, the real packed latent and image bottlenecks, the real MWNOT graph
and sequence modules, distribution distillation, the real TokenFilter policy
head, a deterministic policy-gradient control, and the real Feistel objective.
Every iterative probe rejects non-finite values and insufficient loss decline.

These are trajectory smoke tests, not scale claims. They catch detached
gradients, incompatible shapes, unstable objectives, broken reconstruction,
and objectives that cannot learn their tiny control. Full checkpoint training
still needs the documented data, shared model, service, and hardware, followed
by held-out evaluation. Reports and checkpoints go to ignored `outputs/` or
`artifacts/` directories.
