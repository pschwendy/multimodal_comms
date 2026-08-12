# Collab-Overcooked experiments

The Python environment, planners, prompts, browser renderer, tokenizer,
and application are under `src/multimodal_comms/apps/collab_overcooked`.
Communication can reset per episode or timestep; these are distinct recorded
adapter variants.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENAI_API_KEY=...
python -m multimodal_comms.apps.collab_overcooked.main \
  --order baked_bell_pepper --horizon 120 --episode 1 \
  --compressor identity --channel-scope episode \
  --log_dir outputs/collab_overcooked/identity
```

For a local model, pass `--gpt_model`, `--model_dirname`, and
`--local_server_api`. For DeepSeek, set `DEEPSEEK_API_KEY`. The four
`run_collab_*.sh` programs contain the complete task/condition matrices;
`run_eval_pipeline.sh` runs native evaluation for each completed run.

Native interpretation includes success, task-efficiency score/F1, semantic
similarity, redundancy, and collaboration. Do not compare traffic reductions
without also checking success and error counts. Smoke coverage loads a real
Overcooked layout and performs a tiny deterministic environment transition;
network-backed LLM behavior remains opt-in.

Run the complete condition matrix and native evaluation with:

```bash
bash experiments/collab_overcooked/run_full.sh
```

For a one-order check first, set `QUICK=1`, `ORDER`, `MODEL`, and `METHOD`.
The full path executes episodes before native evaluation; it never substitutes
a generic text score for environment success.
