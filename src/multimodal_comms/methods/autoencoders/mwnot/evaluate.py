from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .configs import MWNOTConfig
from .dataset import WMGMDataset, WMGMDatasetConfig, collate_wmgm
from .losses import MWNOTLoss, metrics
from .model import MWNOTModel
from .utils import default_device, set_seed
from .wmgm import CorruptionConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MWNOT across graph sizes and corruptions.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--K-list", type=int, nargs="+", default=[3, 4])
    p.add_argument("--missing-edge-list", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    p.add_argument("--spurious-edge-list", type=float, nargs="+", default=[0.0])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = MWNOTConfig(**ckpt["model_config"])
    model = MWNOTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    criterion = MWNOTLoss(cfg.p_loss_weight, cfg.l_loss_weight)

    for K in args.K_list:
        for miss in args.missing_edge_list:
            for spur in args.spurious_edge_list:
                corr = CorruptionConfig(missing_edge_prob=miss, spurious_edge_prob=spur)
                ds = WMGMDataset(WMGMDatasetConfig(args.samples, cfg.M, K, corruption=corr), seed=args.seed + K)
                loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_wmgm)
                totals = {"loss": 0.0, "bce": 0.0, "kl": 0.0, "p_mae": 0.0, "l_mae": 0.0}
                count = 0
                with torch.no_grad():
                    for batch in loader:
                        A, mask = batch["A"].to(device), batch["mask"].to(device)
                        p1, l1 = batch["p1"].to(device), batch["l1"].to(device)
                        out = model(A, mask)
                        loss = criterion(out["p_logits"], out["l_logits"], p1, l1)
                        m = metrics(out["p_logits"], out["l_logits"], p1, l1)
                        b = A.shape[0]
                        totals["loss"] += loss["loss"].item() * b
                        totals["bce"] += loss["bce"].item() * b
                        totals["kl"] += loss["kl"].item() * b
                        totals["p_mae"] += m["p_mae"] * b
                        totals["l_mae"] += m["l_mae"] * b
                        count += b
                print(f"K={K} missing_edge={miss} spurious={spur} " + str({k: v / count for k, v in totals.items()}))


if __name__ == "__main__":
    main()
