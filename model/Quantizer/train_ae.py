from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from imu_OVHAR import OVHARDataset, VQVAEIMU, device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent.parent / "dataset"

    parser = argparse.ArgumentParser(description="Train VQ-VAE on OVHAR IMU windows.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset, help="Root folder containing Participant_*")
    parser.add_argument("--seq-length", type=int, default=48, help="Timesteps per training window.")
    parser.add_argument("--stride", type=int, help="Stride between windows (default: seq-length).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-hiddens", type=int, default=128)
    parser.add_argument("--num-residual-layers", type=int, default=2)
    parser.add_argument("--num-residual-hiddens", type=int, default=32)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-files", type=int, help="Limit number of Sensor CSVs to load.")
    parser.add_argument("--save-path", type=Path, default=Path("train_ae_ovhar.pt"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, float]:
    dataset = OVHARDataset(
        root=args.dataset_root,
        seq_length=args.seq_length,
        stride=args.stride,
        max_files=args.max_files,
    )
    if len(dataset) < 2:
        raise ValueError("Not enough windows to split into train/val sets.")

    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, dataset.data_variance


def train_epoch(
    model: VQVAEIMU,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    data_variance: float,
    device_: torch.device,
) -> tuple[float, float]:
    model.train()
    recon_running, perplexity_running = 0.0, 0.0
    for batch, _ in loader:
        batch = batch.to(device_)
        optimizer.zero_grad()
        vq_loss, recon, perplexity = model(batch)
        recon_error = F.mse_loss(recon, batch) / data_variance
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()

        recon_running += recon_error.item()
        perplexity_running += perplexity.item()

    steps = len(loader)
    return recon_running / steps, perplexity_running / steps


@torch.no_grad()
def evaluate(
    model: VQVAEIMU, loader: DataLoader, data_variance: float, device_: torch.device
) -> tuple[float, float]:
    model.eval()
    recon_running, perplexity_running = 0.0, 0.0
    for batch, _ in loader:
        batch = batch.to(device_)
        vq_loss, recon, perplexity = model(batch)
        recon_error = F.mse_loss(recon, batch) / data_variance
        recon_running += (recon_error + vq_loss).item()
        perplexity_running += perplexity.item()

    steps = len(loader)
    return recon_running / steps, perplexity_running / steps


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device_ = device()
    print(f"Using device: {device_}")

    train_loader, val_loader, data_variance = build_dataloaders(args)
    print(f"Loaded {len(train_loader.dataset)} train and {len(val_loader.dataset)} val windows.")

    model = VQVAEIMU(
        num_hiddens=args.num_hiddens,
        num_residual_layers=args.num_residual_layers,
        num_residual_hiddens=args.num_residual_hiddens,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
    ).to(device_)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_recon, train_perplex = train_epoch(model, train_loader, optimizer, data_variance, device_)
        val_loss, val_perplex = evaluate(model, val_loader, data_variance, device_)
        print(
            f"Epoch {epoch:02d} | train recon {train_recon:.4f} perplexity {train_perplex:.2f} | "
            f"val loss {val_loss:.4f} perplexity {val_perplex:.2f}"
        )

    torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.save_path)
    print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
