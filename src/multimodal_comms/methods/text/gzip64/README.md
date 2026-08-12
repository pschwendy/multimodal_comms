# gzip64

Serializes text through gzip and base64, then reverses it before the receiver prompt. It is byte-lossless but base64 overhead can make short messages larger. Experiment: compression sweeps.
