# Project summary

## Goal

This project studies how AI agents can communicate useful information through
a constrained channel. The central question is not simply whether a message
can be made shorter. It is whether a sender can encode its information into a
small transmission, allow a receiver to reconstruct or use that information,
and preserve the team's ability to complete a task.

The current focus is learned compress–decompress communication. A message is
encoded into a compact latent representation, transmitted with explicit
traffic accounting, and decoded back into text or another usable modality.
This makes it possible to study the full tradeoff between:

- packet size and communication cost;
- reconstruction and semantic fidelity;
- downstream task success;
- the number of messages sharing a packet;
- interference between packed messages; and
- receiver isolation and information leakage.

The goal is therefore end-to-end communication efficiency, not compression in
isolation. A method is useful only if the reconstructed information remains
useful to the receiving agent.

## Current compress–decompress approaches

The baseline learned codec is a continuous-latent text autoencoder. It maps a
variable-length message to a fixed number of language-model hidden vectors and
reconstructs the message autoregressively from those vectors. The encoder and
decoder share the same causal language model. This provides the basic
message-to-latents-to-message path used by the newer experiments.

The MWNOT autoencoder replaces fixed-position latent sampling with a learned
multiwavelet sequence generator. Instead of selecting only a few hidden-state
positions, it allows the complete hidden-state sequence to contribute to the
fixed-size code. It is intended to test whether a more structured encoder can
preserve more information at the same latent budget.

The packing work adds a learned nested bottleneck to these latent codes. A
single checkpoint can emit codes at several widths, allowing experiments to
measure reconstruction quality as more messages are placed in a fixed-size
packet. Three packet constructions are kept distinct:

- Block packing assigns disjoint capacity to each message and is exact up to a
  hard slot limit.
- Rotor packing applies an invertible rotation to block packing. It mixes
  coordinates but has the same fidelity and capacity as block packing.
- Frame packing permits overlapping, overloaded packets, with reconstruction
  error that grows through crosstalk.

The superposition and CryptAE experiments bind latent messages with
slot-specific transforms, combine them into one packet, and unbind the target
slot before decoding. The decoder is trained under a load curriculum so it can
tolerate the resulting interference. These experiments measure reconstruction
versus load as well as receiver isolation and known leakage modes. Row-specific
keys and fresh packet nonces address identified failures of shared or reused
linear transforms, but the experiments should not be interpreted as a general
proof of cryptographic security.

The same packet interface also supports learned image bottlenecks and mixed
text–image packets. This allows the project to ask whether one communication
system can carry different modalities while retaining modality-appropriate
reconstruction metrics.

## Evaluation

Every learned method is evaluated at three levels. Training checks establish
that its objective is finite and improves on small learnable problems.
Checkpoint validation measures reconstruction, semantic similarity, capacity,
crosstalk, quantization, or leakage as appropriate. Finally, the trained method
is inserted into information-asymmetry benchmarks, where agents must exchange
private facts or observations to solve a shared task.

The principal benchmark comparison is against uncompressed identity
communication under the same model, task subset, seed, rounds, and channel-view
policy. Results should report both communication traffic and native task
quality. A smaller packet is not a success if reconstruction errors prevent the
agents from solving the task.

HiddenBench, COMMA, Collab-Overcooked, and iAgents provide complementary task
settings. They are evaluation environments rather than owners of training or
method code. Training data, optimization, checkpoint validation, reusable
codecs, and complete experiment trajectories remain separate so a codec can be
tested consistently across applications.

## Supporting and diagnostic approaches

The repository also contains extractive selectors, abstractive rewriters,
predictive-difference methods, conventional lossless codecs, and semantic
certification. These provide useful baselines and alternative ways to reduce
traffic, but they are secondary to the current learned compress–decompress
direction.

Compressed sensing and low-rank methods are retained as diagnostic studies of
representation structure. Compressed sensing works when the signal is sparse
in a shared basis, as image blocks often are. It should not be expected to
reconstruct generic dense embedding matrices; PCA or SVD can be a better model
when a corpus is low-rank instead. This negative result helps distinguish a
working reconstruction algorithm from an incorrect assumption about the data.

The full training and evaluation paths are documented in
[`experiments/`](experiments/README.md), beginning with the
[autoencoder](experiments/autoencoders/README.md),
[MWNOT](experiments/mwnot/README.md),
[packing](experiments/packing/README.md), and
[CryptAE](experiments/crypt_ae/README.md) experiments.
