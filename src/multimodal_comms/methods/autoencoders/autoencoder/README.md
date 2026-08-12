# autoencoder

Samples a fixed number of hidden-state positions as latent tokens and autoregressively reconstructs text. Train with `pretrain_autoencoder`; the decode-side frozen embeddings retain the required `.clone()` fix. Evaluate qualitatively and in HiddenBench.
