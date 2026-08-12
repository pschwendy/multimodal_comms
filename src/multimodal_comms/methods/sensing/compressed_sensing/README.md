# compressed_sensing

Compressed sensing transmits random linear measurements and reconstructs a
signal whose coefficients are sparse in a shared dictionary. The public codec
uses OMP; the larger experiment implementation also retains Lasso. Ridge is an
explicit minimum-norm projection baseline, not sparse recovery.

For image blocks, use `make_dictionary(..., spatial_shape=(height, width))` to
build the separable 2-D DCT basis. Smooth image blocks are compressible there.
Do not assume an arbitrary embedding vector is sparse: the behavioral check
deliberately shows OMP failing on dense embeddings, then shows PCA succeeding
when a different low-rank corpus assumption actually holds.

Implementation: `methods/sensing/compressed.py` and
`methods/sensing/dictionary.py`. Experiments and interpretation:
`experiments/compressed_sensing/README.md`.
