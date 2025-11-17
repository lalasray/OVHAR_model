from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from enc_dec import IMUDecoder, IMUEncoder
from vector_quant import VectorQuantizer, VectorQuantizerEMA


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OVHARDataset(Dataset):
    """Build sliding windows from every Sensor CSV under the dataset root."""

    def __init__(
        self,
        root: Path,
        seq_length: int = 30,
        stride: int | None = None,
        columns: Sequence[str] = ("ax", "ay", "az"),
        max_files: int | None = None,
    ) -> None:
        self.seq_length = seq_length
        self.stride = stride or seq_length
        self.columns = list(columns)
        self.windows: list[np.ndarray] = []
        self.data_variance: float = 1.0

        csv_paths = sorted(root.rglob("Sensor*.csv"))
        if max_files is not None:
            csv_paths = csv_paths[:max_files]
        if not csv_paths:
            raise FileNotFoundError(f"No Sensor*.csv files found under {root}")

        sq_sum = 0.0
        value_count = 0
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if not all(col in df.columns for col in self.columns):
                missing = [c for c in self.columns if c not in df.columns]
                raise ValueError(f"{csv_path} missing columns {missing}")

            # Center the raw signals per file to reduce bias across sensors.
            values = df[self.columns].to_numpy(dtype=np.float32)
            values -= values.mean(axis=0, keepdims=True)

            for start in range(0, len(values) - seq_length + 1, self.stride):
                window = values[start : start + seq_length].T  # (C, T)
                self.windows.append(window)
                sq_sum += float(np.sum(window ** 2))
                value_count += window.size

        if value_count > 0:
            self.data_variance = sq_sum / value_count

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.windows[idx]), 0


class VQVAEIMU(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0.0,
    ) -> None:
        super().__init__()
        self._encoder = IMUEncoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = IMUDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._encoder(x)
        z = self._pre_vq_conv(z).unsqueeze(-1)
        vq_loss, quantized, perplexity, _ = self._vq(z)
        quantized = quantized.squeeze(-1)
        x_recon = self._decoder(quantized)
        return vq_loss, x_recon, perplexity


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent.parent / "dataset"

    parser = argparse.ArgumentParser(description="Train the IMU VQ-VAE on OVHAR CSV data.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset, help="Root folder that holds Participant_*")
    parser.add_argument("--seq-length", type=int, default=48, help="Number of timesteps per training window.")
    parser.add_argument("--stride", type=int, help="Stride between windows (default: same as seq-length).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-hiddens", type=int, default=128)
    parser.add_argument("--num-residual-layers", type=int, default=2)
    parser.add_argument("--num-residual-hiddens", type=int, default=32)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of windows reserved for validation.")
    parser.add_argument("--max-files", type=int, help="Limit number of Sensor CSVs to load (for quick tests).")
    parser.add_argument("--save-path", type=Path, default=Path("imu_ovhar.pt"), help="Checkpoint path.")
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
    generator = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)
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
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

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
