# Curated findings

Raw reports and transcripts are intentionally excluded. The concise findings
below are tied to the proofs and experiment configurations in this repository:

- Incremental predictive encoding and replay must use the same numerical path;
  batched versus cached bf16 argmax can diverge near ties and break losslessness.
- A single orthogonal key reused across all rows preserves the plaintext Gram
  matrix. Independent row keys remove that invariant (subject to the stated
  key-generation assumptions).
- Rotor packing is an invertible transform of block packing, so its fidelity and
  hard capacity equal the block reference. Frame packing is the overloadable,
  interference-bearing alternative.
- Learned autoencoder training must clone frozen embedding outputs when the
  decode-side computation requires gradients.

Untraceable score claims are out of scope.

## Compact benchmark observations

A COMMA DeepSeek sweep covered six ATM and six Telehealth tasks.
Identity solved 4/6 Telehealth items; saliency solved 3/6 while reducing prompt
tokens on both task types; the representation-match rewriter solved 1/6.
ATM was unsolved under identity too because a text-only Solver could not inspect
the rendered board, so ATM measurements are useful for traffic but not method
accuracy. Full configuration and caveats are in `experiments/comma/README.md`.

No reported score is an acceptance threshold. Smoke tests validate code and
schemas with deterministic fixtures, not score reproduction.

An iAgents Needle sweep used 25 items. Identity reported 17/25;
representation-match best-of-k reported 18/25 with 26.3% lower transmitted
characters, while saliency saved 63.6% but fell to 6/25. The iAgents experiment
README records the other conditions and requires independent regrading.
