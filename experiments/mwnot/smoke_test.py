from __future__ import annotations

import torch

from multimodal_comms.methods.autoencoders.mwnot.configs import MWNOTConfig
from multimodal_comms.methods.autoencoders.mwnot.dataset import collate_wmgm
from multimodal_comms.methods.autoencoders.mwnot.model import MWNOTModel
from multimodal_comms.methods.autoencoders.mwnot.wmgm import generate_wmgm_graph, sample_base_params


def main() -> None:
    M, K = 3, 3
    p1, l1 = sample_base_params(M)
    graph = generate_wmgm_graph(p1, l1, K=K, sort_nodes=True)
    batch = collate_wmgm([graph])
    cfg = MWNOTConfig(M=M, embed_dim=64, num_heads=4, num_layers=1, wavelet_levels=2, poly_order=2)
    model = MWNOTModel(cfg)
    pred = model.predict(batch["A"], batch["mask"])
    print("A shape:", tuple(batch["A"].shape))
    print("p1 prediction shape:", tuple(pred["p1"].shape), "range:", float(pred["p1"].min()), float(pred["p1"].max()))
    print("l1 prediction shape:", tuple(pred["l1"].shape), "sum:", float(pred["l1"].sum(-1).item()))
    print("target p1 shape:", tuple(batch["p1"].shape), "target l1:", batch["l1"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
