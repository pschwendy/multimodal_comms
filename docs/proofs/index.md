# Proof and correction notes

## Exact block/rotor equivalence

Let `x` concatenate all occupied `d`-wide blocks in a `P`-vector and let `R` be
orthogonal. Block emits `x`; Rotor emits `xR`. Rotor recovery first computes
`(xR)Rᵀ = x` and slices the target block. Both methods therefore recover the
same code up to floating-point roundoff and both have capacity `floor(P/d)`.
An invertible transform does not add information capacity.

## Shared-row-key Gram leak

For a latent matrix `Z` and one reused orthogonal key `Q`, ciphertext `C=ZQ`
satisfies `CCᵀ=ZQQᵀZᵀ=ZZᵀ`. Every row norm and pairwise inner product is
visible. If row `i` instead uses an independent `Qᵢ`, an off-diagonal term is
`zᵢ Qᵢ Qⱼᵀ zⱼᵀ`; the plaintext inner product is no longer invariant. This is a
regression property, not by itself a complete security proof.

## Nonces

Long-term private keys must derive a fresh per-packet transform from a unique
nonce. `SecureBroadcast` rejects local nonce reuse. Distributed deployments
must additionally coordinate nonce uniqueness across processes.

These notes supersede claims that a shared row key was IND-CPA secure or that
Rotor provided information capacity beyond concatenation.

