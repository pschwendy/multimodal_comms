# Cryptographic autoencoder

CryptAE combines a continuous-latent text autoencoder with slot-specific
binding, one-time packet nonces, and a decoder trained to tolerate
superposition crosstalk. It is evaluated as both a communication method and a
security construction.

## Full trajectory

```bash
BENCH_CONFIG=experiments/hiddenbench/configs/config.example.yaml \
DEVICE=cuda:0 BASE_STEPS=3000 STEPS=3000 \
bash experiments/crypt_ae/run_full.sh
```

The stages are `data`, `base`, `train`, `validate`, and `benchmark`. `base`
trains the single-message autoencoder. `train` starts from that checkpoint and
uses a load curriculum to train keyed superposition. `validate` measures text
fidelity, intended-recipient similarity, unintended-slot similarity, and the
Gram-leak regression over loads. `benchmark` compares `superpose` with
`identity` on the same HiddenBench sample.

Use row-specific keys and a fresh nonce for each packet when confidentiality is
required. A shared row transform leaks pairwise Gram structure; a reused
linear key is vulnerable to known-plaintext recovery. The regression
conditions remain in the evaluator so those failures stay visible.

Interpret reconstruction versus load together with privacy measurements and
task success. A packet may have a fixed wire shape while semantic fidelity
degrades as crosstalk increases. `MAX_SLOTS`, `LOADS`, `KEY_MODE`, model paths,
and output paths can all be overridden through environment variables.
