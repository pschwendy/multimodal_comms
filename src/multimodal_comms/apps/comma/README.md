# COMMA application

COMMA is a desktop benchmark for collaborative multimodal puzzle solving.
Two agents receive complementary views and communicate to complete tasks such
as ATM, wire, maze, and Telehealth puzzles.

Use the repository-level environment and launch from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m multimodal_comms.apps.comma.run_comma \
  --puzzle_config src/multimodal_comms/apps/comma/config/puzzles_small.json \
  --model_config src/multimodal_comms/apps/comma/config/deepseek_model.json \
  --save_folder outputs/comma
```

On a headless Linux host, prefix the command with `xvfb-run -a`. Linux- and
Windows-specific dependencies are listed in `requirements-linux.txt` and
`requirements-windows.txt`. The Telehealth task requires an external PAD-UFES
image directory configured with `COMMA_PAD_UFES_DIR`; that dataset is not part
of the repository.

Provider credentials belong in environment variables. Do not put real keys in
the example JSON configuration. The full experimental protocol and result
interpretation are in `experiments/comma/README.md`.
