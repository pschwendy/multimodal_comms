# Learned packing

This experiment trains a nested bottleneck so latent messages can occupy
different code widths, then evaluates Block, Frame, and Rotor packing at
matched capacity. These packers are distinct: Block is exact within its slot
budget, Rotor permutes the same hard-capacity layout, and Frame can overload a
packet at the cost of interference.

```bash
BASE_CHECKPOINT=outputs/models/autoencoder/final \
DATA_DIR=outputs/data/fineweb_ae DEVICE=cuda:0 \
bash experiments/packing/run_full.sh
```

Stages are `train`, `validate`, and `multimodal`. Training starts from a
sampled-latent checkpoint. Validation reports fidelity across message counts,
quantization settings, overload, and crosstalk. The optional multimodal stage
trains the image bottleneck from `IMAGE_DIR` and evaluates mixed packets.

Interpret all curves at the same total packet dimension and bit depth.
RotorPacker does not create additional degrees of freedom; it should match
BlockPacker fidelity up to capacity. Frame overload is useful only when its
task-level degradation is acceptable.
