from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from imu_OVHAR import OVHARDataset, VQVAEIMU, device


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent.parent / "dataset"
    parser = argparse.ArgumentParser(description="Infer/reconstruct a window from OVHAR using a trained checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--seq-length", type=int, default=None, help="If omitted, use seq_length from checkpoint args.")
    parser.add_argument("--stride", type=int, default=None, help="If omitted, use stride from checkpoint args.")
    parser.add_argument("--max-files", type=int, help="Limit Sensor CSVs to load.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Which window to visualize.")
    parser.add_argument("--checkpoint", type=Path, default=Path("train_ae_ovhar.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_ = device()
    print(f"Using device: {device_}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    seq_length = args.seq_length if args.seq_length is not None else ckpt_args.get("seq_length", 24)
    stride = args.stride if args.stride is not None else ckpt_args.get("stride", 1)
    ds = OVHARDataset(root=args.dataset_root, seq_length=seq_length, stride=stride, max_files=args.max_files)
    if not (0 <= args.sample_idx < len(ds)):
        raise IndexError(f"sample-idx {args.sample_idx} out of range (0..{len(ds)-1})")

    model = VQVAEIMU(
        num_hiddens=ckpt_args.get("num_hiddens", 128),
        num_residual_layers=ckpt_args.get("num_residual_layers", 2),
        num_residual_hiddens=ckpt_args.get("num_residual_hiddens", 32),
        num_embeddings=ckpt_args.get("num_embeddings", 512),
        embedding_dim=ckpt_args.get("embedding_dim", 64),
        commitment_cost=ckpt_args.get("commitment_cost", 0.25),
        decay=ckpt_args.get("decay", 0.99),
    ).to(device_)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    window, (label_id, label_text) = ds[args.sample_idx]
    with torch.no_grad():
        _, recon, _ = model(window.unsqueeze(0).to(device_))
    recon_np = recon.squeeze(0).cpu().numpy()

    time_axis = range(window.shape[-1])
    fig, axs = plt.subplots(window.shape[0], 1, figsize=(10, 6), sharex=True)
    if window.shape[0] == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.plot(time_axis, window[i].numpy(), label="Original", linestyle="--", alpha=0.7)
        ax.plot(time_axis, recon_np[i], label="Reconstructed", alpha=0.9)
        ax.set_ylabel(f"ch {i}")
        ax.legend(loc="upper right")
    axs[-1].set_xlabel("Time steps")
    fig.suptitle(f"Window {args.sample_idx} | label_id={label_id} | text='{label_text}'")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
