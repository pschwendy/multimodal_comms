from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .configs import MWNOTConfig, TrainConfig
from .dataset import WMGMDataset, WMGMDatasetConfig, collate_wmgm
from .losses import MWNOTLoss, metrics
from .model import MWNOTModel
from .utils import default_device, linear_warmup_cosine, save_json, set_seed


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MWNOT on synthetic WMGM graphs.")
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--train-samples", type=int, default=1024)
    p.add_argument("--val-samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=5)
    p.add_argument("--poly-order", type=int, default=3)
    p.add_argument("--wavelet-levels", type=int, default=3)
    p.add_argument("--lifting", choices=["legendre", "linear"], default="legendre")
    p.add_argument("--disable-wavelets", action="store_true")
    p.add_argument("--no-sort", action="store_true")
    p.add_argument("--min-nodes", type=int, default=None)
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--lambda-l", type=float, default=1.0)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool, grad_clip: float = 1.0) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "bce": 0.0, "kl": 0.0, "p_mae": 0.0, "l_mae": 0.0}
    count = 0
    for batch in loader:
        A = batch["A"].to(device)
        mask = batch["mask"].to(device)
        p1 = batch["p1"].to(device)
        l1 = batch["l1"].to(device)
        with torch.set_grad_enabled(train):
            out = model(A, mask)
            loss_dict = criterion(out["p_logits"], out["l_logits"], p1, l1)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        m = metrics(out["p_logits"].detach(), out["l_logits"].detach(), p1, l1)
        b = A.shape[0]
        totals["loss"] += loss_dict["loss"].detach().item() * b
        totals["bce"] += loss_dict["bce"].item() * b
        totals["kl"] += loss_dict["kl"].item() * b
        totals["p_mae"] += m["p_mae"] * b
        totals["l_mae"] += m["l_mae"] * b
        count += b
    return {k: v / max(1, count) for k, v in totals.items()}


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = default_device(args.device)
    model_cfg = MWNOTConfig(
        M=args.M,
        patch_size=args.patch_size,
        poly_order=args.poly_order,
        lifting=args.lifting,
        use_wavelets=not args.disable_wavelets,
        wavelet_levels=args.wavelet_levels,
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        sort_nodes=not args.no_sort,
        l_loss_weight=args.lambda_l,
    )
    train_cfg = TrainConfig(
        seed=args.seed,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        data_K=args.K,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        lambda_l=args.lambda_l,
        device=args.device,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_json(os.path.join(args.checkpoint_dir, "model_config.json"), asdict(model_cfg))
    save_json(os.path.join(args.checkpoint_dir, "train_config.json"), asdict(train_cfg))

    train_ds = WMGMDataset(WMGMDatasetConfig(args.train_samples, args.M, args.K, args.min_nodes, args.max_nodes), seed=args.seed)
    val_ds = WMGMDataset(WMGMDatasetConfig(args.val_samples, args.M, args.K, args.min_nodes, args.max_nodes), seed=args.seed + 100000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_wmgm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wmgm)

    model = MWNOTModel(model_cfg).to(device)
    criterion = MWNOTLoss(model_cfg.p_loss_weight, model_cfg.l_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=train_cfg.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: linear_warmup_cosine(step, total_steps, train_cfg.warmup_steps)
    )

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, True, train_cfg.grad_clip)
        va = run_epoch(model, val_loader, criterion, None, None, device, False, train_cfg.grad_clip)
        print(f"epoch {epoch:03d} train {tr} val {va}")
        state = {"model": model.state_dict(), "model_config": asdict(model_cfg), "epoch": epoch, "val": va}
        torch.save(state, os.path.join(args.checkpoint_dir, "last.pt"))
        if va["loss"] < best:
            best = va["loss"]
            torch.save(state, os.path.join(args.checkpoint_dir, "best.pt"))


if __name__ == "__main__":
    main()
