# Sentence selectors

Selector methods keep message spans predicted to matter to a receiver. The
basic supervised trajectory labels sentences from completed training-task
discussions, fits a compact classifier, then evaluates that classifier inside
the asynchronous discussion loop.

The launcher creates a deterministic, disjoint task split, generates identity
discussions for both parts, builds labels only from the training split, trains,
validates, and runs the learned selector on the evaluation split:

```bash
BENCH_CONFIG=experiments/hiddenbench/configs/config.example.yaml \
bash experiments/selectors/run_full.sh
```

Stages are `harvest`, `data`, `train`, `validate`, and `benchmark`. The data builder
excludes every task named by `EVAL_REPORT`, so evaluation tasks cannot provide
training labels. Validation runs the deterministic classifier progress probe.
Benchmarking compares `learned` with `identity` using the same task sample.

Representation-match, counterfactual, VIB, rewriter, and token-filter
variants use the adjacent harvest/train programs in `training/`. They follow
the same split discipline but require a representation or receiver service as
documented by each program's `--help` and module docstring. Judge a selector by
native task quality and transmitted traffic, not by its keep rate alone.
