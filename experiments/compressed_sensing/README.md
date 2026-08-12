# Compressed sensing

These experiments test compressed sensing where its sparsity assumptions can
and cannot be justified. Images are positive controls because local blocks are
often sparse in a DCT or wavelet basis. Generic language-model embeddings are
a negative control: they are usually dense in the tested bases, so sparse
recovery should not be expected to reconstruct them.

```bash
bash experiments/compressed_sensing/run_full.sh
```

Stages are `assumptions`, `images`, and `embeddings`. The assumption stage is
deterministic and checks sparse-vector recovery, natural-image block PSNR/SSIM,
dense embedding failure, and low-rank SVD behavior. The image stage runs the
partial-Fourier/TV and block-DCT/OMP pipelines. The embedding stage downloads
the configured model and compares reconstruction from identical measurements.

The current smoke thresholds require sparse DCT recovery below 0.02 relative
error and natural-image block recovery above 22 dB PSNR and 0.65 SSIM. Dense
Gaussian embeddings must not be reported as a CS success; the deterministic
control has OMP relative error above 0.9. For real embeddings, compare OMP
with ridge/pseudoinverse and PCA/SVD at the same measurement budget.

If image recovery succeeds while embedding recovery fails, the implementation
is behaving correctly and the embedding sparsity model is wrong. More OMP
iterations do not repair a mismatched prior. Model downloads, arrays, plots,
and reports are written beneath ignored output directories.
