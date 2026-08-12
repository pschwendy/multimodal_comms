# iAgents experiments

The Flask application, templates, static files, prompts, backends,
runtime, and sanitized configuration live under
`src/multimodal_comms/apps/iagents`. Dataset generators for FriendsTV,
Needle-in-the-Persona, and Schedule are retained here with their database
importers.

Offline evaluation needs JSONL records containing a task prompt, answer, and
the persona/context fields used by `needle_common.py`:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export DEEPSEEK_API_KEY=...
python -m experiments.iagents.programs.run_offline_eval \
  --dataset /path/eval.jsonl --condition identity --limit 1 \
  --out outputs/iagents/identity.json
python -m experiments.iagents.programs.regrade \
  outputs/iagents/identity.json --out outputs/iagents/identity_regraded.json
```

`run_sweep.sh` executes all five communication conditions. The learned chain
is under `training/benchmark_adaptation/iagents/`: harvest ->
`build_repmatch_dataset` -> `train_repmatch_selector`, plus harvest ->
`train_repmatch_grpo`.

For an identity/method pair followed by independent regrade and aggregation:

```bash
DATASET=/path/eval.jsonl METHOD=saliency METHOD_PARAMS="--param rate=0.4" \
bash experiments/iagents/run_full.sh
```

To run the web app, copy the sanitized `config/global.yaml`, set database and
provider values through the documented environment/config mechanism, create
the database with `python -m multimodal_comms.apps.iagents.create_database`,
then start `python -m multimodal_comms.apps.iagents.app`. Offline smoke never
requires MySQL, an API, or a live model server. Report both original answer
grading and independent regrade results.

One 25-item Needle sweep reported identity at 17/25 with 91,862
transmitted characters. Representation-match best-of-k reported 18/25 with
26.3% fewer characters; the learned selector reported 16/25 with 23.9% fewer;
saliency was much smaller but fell to 6/25. These values describe one recorded
DeepSeek configuration and must be interpreted alongside the independent
regrade rather than treated as regression thresholds.
