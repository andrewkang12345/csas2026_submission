#!/usr/bin/env python3
"""
train_with_synth_wandb.py

W&B-runnable trainer for ValueTransformer using real + (subsampled) synthetic data,
with validation on REAL only.

Key additions:
  - W&B init + config capture
  - logs train/val loss, LR, epoch time, grad norm
  - saves best checkpoint and logs as W&B artifact
  - NEW: --synth_frac and --synth_max to control how much synthetic data is used
         (default: synth_frac=0.25, capped by synth_max=200000)

Rationale for default synth usage:
  - In practice, synthetic distributions can dominate if you take all synth rows.
  - 25% provides regularization and broader coverage while still letting real data steer.
  - A hard cap avoids accidental huge synthetic runs.

You can sweep over synth_frac to empirically pick the best mix.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

import wandb

from dataset import ValueDataset, NUM_STONES
from model import ValueTransformer


def _log(msg: str, log_file: Optional[Path]):
    print(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def _subsample_dataset(ds: Dataset, frac: float, max_n: int, seed: int) -> Dataset:
    """
    Returns a Subset(ds) with size = min(max_n, round(frac * len(ds))).
    If frac>=1 and max_n>=len(ds), returns ds unchanged.
    """
    n = len(ds)
    if n == 0:
        return ds
    frac = float(frac)
    if frac <= 0.0 or max_n == 0:
        return Subset(ds, [])
    k = int(round(frac * n))
    if max_n is not None and max_n > 0:
        k = min(k, int(max_n))
    k = max(0, min(k, n))
    if k == n:
        return ds
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=k, replace=False)
    idx = np.asarray(idx, dtype=np.int64).tolist()
    return Subset(ds, idx)


def save_ckpt(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def train(args):
    log_path = Path(args.log_file) if args.log_file else None
    out_path = Path(args.out)

    # -----------------
    # W&B init
    # -----------------
    wandb_mode = "offline" if args.wandb_offline else "online"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.wandb_name if args.wandb_name else None,
        group=args.wandb_group if args.wandb_group else None,
        job_type=args.wandb_job_type,
        tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
        mode=wandb_mode,
        config=vars(args),
        resume="allow" if args.wandb_resume else None,
    )

    # -----------------
    # Datasets
    # -----------------
    real_ds = ValueDataset(args.stones_csv, args.ends_csv)
    synth_ds_full = ValueDataset(args.synth_stones_csv, args.synth_ends_csv)

    # Decide synth usage (default 25%, capped)
    synth_ds = _subsample_dataset(
        synth_ds_full,
        frac=args.synth_frac,
        max_n=args.synth_max,
        seed=args.synth_seed,
    )

    _log(
        f"Real dataset size={len(real_ds)}, Synth dataset size(full)={len(synth_ds_full)}, Synth used={len(synth_ds)}",
        log_path,
    )
    _log(
        f"input_dim={real_ds.input_dim}, cond_dim={real_ds.cond_dim}, num_tasks={real_ds.num_tasks}",
        log_path,
    )

    wandb.config.update(
        {
            "real_size": len(real_ds),
            "synth_size_full": len(synth_ds_full),
            "synth_size_used": len(synth_ds),
            "input_dim": real_ds.input_dim,
            "cond_dim": real_ds.cond_dim,
            "num_tasks": real_ds.num_tasks,
        },
        allow_val_change=True,
    )

    # Split only the real data
    if args.val_split > 0.0:
        val_size = int(len(real_ds) * args.val_split)
        train_size = len(real_ds) - val_size
        g = torch.Generator().manual_seed(args.split_seed)
        real_train_ds, real_val_ds = random_split(real_ds, [train_size, val_size], generator=g)
    else:
        real_train_ds = real_ds
        real_val_ds = None

    # Combine: real_train + (subsampled) synth
    train_ds = ConcatDataset([real_train_ds, synth_ds])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            real_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if real_val_ds is not None
        else None
    )

    # -----------------
    # Device / model
    # -----------------
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    _log(f"Using device: {device}", log_path)
    wandb.config.update({"device": str(device)}, allow_val_change=True)

    model = ValueTransformer(
        input_dim=real_ds.input_dim,
        cond_dim=real_ds.cond_dim,  # expects 3 in your revamped dataset
        hidden_dim=args.hidden_dim,
        num_stones=NUM_STONES,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    if args.wandb_watch:
        wandb.watch(model, log="gradients", log_freq=args.wandb_watch_freq)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # -----------------
    # Resume checkpoint
    # -----------------
    start_epoch = 1
    best_loss = float("inf")

    if args.resume and os.path.exists(args.resume):
        _log(f"Resuming from checkpoint: {args.resume}", log_path)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_loss" in ckpt:
            best_loss = float(ckpt["best_loss"])
    elif args.resume:
        _log(f"WARNING: --resume path {args.resume} does not exist. Starting from scratch.", log_path)

    _log(
        f"Starting from epoch {start_epoch} / {args.epochs}. Validation on REAL only (val_split={args.val_split}).",
        log_path,
    )

    # -----------------
    # Training loop
    # -----------------
    global_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_count = 0
        last_grad_norm = float("nan")

        for x, c, y in train_loader:
            x = x.to(device)
            c = c.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, c)
            loss = criterion(pred, y)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            last_grad_norm = float(grad_norm.detach().cpu().item()) if torch.isfinite(grad_norm) else float("nan")

            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            running_count += bs

            if args.log_every_steps > 0 and (global_step % args.log_every_steps == 0):
                wandb.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/grad_norm_step": last_grad_norm,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            global_step += 1

        train_loss = running_loss / max(1, running_count)

        # Validation on REAL only
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for x, c, y in val_loader:
                    x = x.to(device)
                    c = c.to(device)
                    y = y.to(device)
                    pred = model(x, c)
                    v_loss = criterion(pred, y)
                    bs = x.size(0)
                    val_running += float(v_loss.item()) * bs
                    val_count += bs
            val_loss = val_running / max(1, val_count)
            current_metric = float(val_loss)
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_loss={best_loss:.6f}",
                log_path,
            )
        else:
            current_metric = float(train_loss)
            _log(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | best_loss={best_loss:.6f}",
                log_path,
            )

        epoch_time = time.time() - t0

        wandb_metrics = {
            "epoch": epoch,
            "train/loss": float(train_loss),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/grad_norm": float(last_grad_norm),
            "time/epoch_sec": float(epoch_time),
            "best/loss": float(best_loss),
        }
        if val_loss is not None:
            wandb_metrics["val/loss"] = float(val_loss)

        wandb.log(wandb_metrics, step=global_step)

        is_last = (epoch == args.epochs)
        should_check = (epoch % args.save_every == 0) or is_last

        if should_check and current_metric < best_loss:
            best_loss = float(current_metric)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "input_dim": real_ds.input_dim,
                "cond_dim": real_ds.cond_dim,
                "hidden_dim": args.hidden_dim,
                "num_stones": NUM_STONES,
                "num_tasks": real_ds.num_tasks,
                "best_loss": best_loss,
                "args": vars(args),
            }
            save_ckpt(out_path, ckpt)
            _log(f"Saved NEW BEST checkpoint to {out_path} (epoch {epoch}, metric={current_metric:.6f})", log_path)

            art = wandb.Artifact(
                name=args.wandb_artifact_name,
                type="model",
                metadata={
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "input_dim": real_ds.input_dim,
                    "cond_dim": real_ds.cond_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_stones": NUM_STONES,
                    "real_size": len(real_ds),
                    "synth_size_used": len(synth_ds),
                    "synth_frac": float(args.synth_frac),
                    "synth_max": int(args.synth_max),
                },
            )
            art.add_file(str(out_path))
            run.log_artifact(art)

    _log(f"Training finished. Best metric: {best_loss:.6f}. Best checkpoint at {out_path}.", log_path)

    if out_path.exists():
        wandb.save(str(out_path))

    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train value model on real + synthetic (validate on real only) with W&B.")

    # Data
    ap.add_argument("--stones_csv", type=str, default="../2026/Stones.csv")
    ap.add_argument("--ends_csv", type=str, default="../2026/Ends.csv")
    ap.add_argument("--synth_stones_csv", type=str, default="synth_stones.csv")
    ap.add_argument("--synth_ends_csv", type=str, default="synth_ends.csv")

    # Outputs
    ap.add_argument("--out", type=str, default="value_model_synth.pt")
    ap.add_argument("--log_file", type=str, default="train_with_synth.log")
    ap.add_argument("--resume", type=str, default="")

    # Train hyperparams
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every_steps", type=int, default=0)

    # NEW: synth usage controls
    ap.add_argument("--synth_frac", type=float, default=1.0, help="Fraction of synth dataset to use (0..1+).")
    ap.add_argument("--synth_max", type=int, default=200000, help="Hard cap on number of synth samples used.")
    ap.add_argument("--synth_seed", type=int, default=123, help="Seed for synth subsampling.")

    # W&B
    ap.add_argument("--wandb_project", type=str, default="curling-value")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_name", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="")
    ap.add_argument("--wandb_job_type", type=str, default="train_with_synth")
    ap.add_argument("--wandb_tags", type=str, default="")
    ap.add_argument("--wandb_notes", type=str, default="")
    ap.add_argument("--wandb_resume", action="store_true")
    ap.add_argument("--wandb_offline", action="store_true")
    ap.add_argument("--wandb_watch", action="store_true")
    ap.add_argument("--wandb_watch_freq", type=int, default=200)
    ap.add_argument("--wandb_artifact_name", type=str, default="value_model_synth_best")

    args = ap.parse_args()
    train(args)
