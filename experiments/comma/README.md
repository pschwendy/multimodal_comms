# COMMA experiments

The COMMA Tk application and puzzle assets live in
`src/multimodal_comms/apps/comma`. This directory runs puzzle conditions and
sweep aggregation. Optional representation-match data preparation lives under
`training/benchmark_adaptation/comma/`.

```bash
export PYTHONPATH="$PWD/src:$PWD"
export COMMA_COMPRESSOR=identity
python -m multimodal_comms.apps.comma.run_comma \
  --puzzle_config src/multimodal_comms/apps/comma/config/puzzles_small.json \
  --model_config src/multimodal_comms/apps/comma/config/deepseek_model.json \
  --save_folder outputs/comma
```

COMMA is a desktop application. On a headless Linux host, run the command under
`xvfb-run -a`. Model configuration selects the original agent classes and API
credentials come from environment variables. Telehealth's multi-gigabyte
PAD-UFES image corpus is not bundled; set `COMMA_PAD_UFES_DIR` to its image
directory. Manuals, keypad/maze images, sounds, and application artwork are
bundled beside the source.

Run an identity/method pair and aggregate it with:

```bash
METHOD=saliency HEADLESS=1 bash experiments/comma/run_full.sh
```

For a learned COMMA-specific selector, run the harvest/build/train programs in
`training/benchmark_adaptation/comma/`, then pass its checkpoint in the method
configuration used by the application.

Interpret completion per native puzzle. Telehealth additionally reports its
partial-credit score; it must not be reduced to a generic exact-match metric.
Smoke tests validate all program syntax, resource resolution, channel adapter,
and a deterministic puzzle-grade fixture without opening a display or API.

One 12-task DeepSeek sweep found that saliency reduced prompt tokens
on both ATM and Telehealth while staying closest to identity accuracy. ATM was
unsolved even by identity because the text-only solver could not inspect the
rendered board; it is therefore a cost signal, not an accuracy comparison.
Telehealth identity solved 4/6, saliency 3/6, and the abstractive rewriter 1/6.
These are experiment observations, not smoke-test expectations.
