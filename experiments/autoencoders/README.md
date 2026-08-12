# Sampled-latent autoencoder

This experiment compresses a variable-length message into a fixed number of
causal-LM hidden vectors, then reconstructs the message by injecting those
vectors at reserved latent-token positions. The same language model performs
encoding and decoding.

## Full trajectory

`run_full.sh` performs four stages: stream a train/dev text corpus, train the
autoencoder, inspect held-out reconstructions, and compare the checkpoint with
identity on the same HiddenBench tasks.

```bash
BENCH_CONFIG=experiments/hiddenbench/configs/config.example.yaml \
DEVICE=cuda:0 STEPS=3000 TASKS=16 \
bash experiments/autoencoders/run_full.sh
```

Use `STAGE=data`, `train`, `validate`, or `benchmark` to resume one stage.
Override `DATA_DIR`, `MODEL_DIR`, `REPORT_DIR`, `NUM_LATENTS`, and `DEVICE`
without editing the script. `BENCH_CONFIG` must describe a working LLM
provider, but the trained autoencoder checkpoint is passed on the command line.

## What to monitor

Training prints train loss and periodic dev loss/token accuracy. A healthy run
has finite loss, improving dev reconstruction, and a reloadable `final/`
checkpoint. Qualitative validation catches degenerate outputs that token
accuracy can hide. The benchmark comparison must report both post-discussion
accuracy/information gain and traffic; reconstruction quality alone does not
establish useful communication.

The decode embeddings are cloned before latent insertion. This is required
when the embedding table is frozen and is covered by the training smoke tests.
Outputs and checkpoints are written beneath ignored `outputs/` paths.
