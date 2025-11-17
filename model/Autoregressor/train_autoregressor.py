from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from model.Autoregressor.model import IMUToTextModel
from model.Quantizer.imu_OVHAR import OVHARDataset, device
from transformers import AutoTokenizer


class LabeledDataset(Dataset):
    """Wrap OVHARDataset and keep only windows that have labels."""

    def __init__(self, base: OVHARDataset):
        self.base = base
        self.indices: List[int] = [i for i, lid in enumerate(base.window_labels) if lid >= 0]
        if not self.indices:
            raise ValueError("No labeled windows found (label_id < 0 for all samples).")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        base_idx = self.indices[idx]
        window, (label_id, label_text) = self.base[base_idx]
        return window, label_id, label_text


def collate_fn_factory(tokenizer: AutoTokenizer):
    def collate_fn(batch):
        windows, _, label_texts = zip(*batch)
        windows_t = torch.stack(windows, dim=0)
        tokenized = tokenizer(
            list(label_texts),
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return windows_t, tokenized["input_ids"], tokenized["attention_mask"], list(label_texts)

    return collate_fn


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent.parent / "dataset"
    parser = argparse.ArgumentParser(description="Train Q-Former + classifier on frozen VQ-VAE encoder.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    default_ckpt = script_dir.parent / "Quantizer" / "train_ae_ovhar.pt"
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_ckpt,
        help="Path to frozen VQ-VAE checkpoint (default: model/Quantizer/train_ae_ovhar.pt)",
    )
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-queries", type=int, default=8)  # retained for backward compatibility (unused for HF)
    parser.add_argument("--qformer-name", type=str, default="Salesforce/blip2-opt-2.7b", help="HF Q-Former backbone.")
    parser.add_argument("--lm-name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="HF causal LM name/path (frozen).")
    parser.add_argument("--max-files", type=int, help="Limit Sensor CSVs to load.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, list[str], LabeledDataset, AutoTokenizer]:
    base = OVHARDataset(
        root=args.dataset_root,
        seq_length=args.seq_length,
        stride=args.stride,
        max_files=args.max_files,
    )
    labeled = LabeledDataset(base)
    val_len = max(1, int(len(labeled) * args.val_split))
    train_len = len(labeled) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(labeled, [train_len, val_len], generator=generator)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = collate_fn_factory(tokenizer)

    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False, collate_fn=collate)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    loader_kwargs["shuffle"] = False
    val_loader = DataLoader(val_ds, **loader_kwargs)
    return train_loader, val_loader, base.id_to_label, labeled, tokenizer


def run_epoch(model, loader, optimizer, device_, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for windows, input_ids, attention_mask, label_texts in tqdm(
        loader, desc="train" if train else "val", leave=False
    ):
        windows = windows.to(device_)
        if train:
            optimizer.zero_grad()
        loss, logits = model(windows, input_ids.to(device_), attention_mask.to(device_))
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * windows.size(0)
        total_samples += windows.size(0)
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device_ = device()
    print(f"Using device: {device_}")

    train_loader, val_loader, id_to_label, labeled_ds, tokenizer = build_loaders(args)
    print(f"Labeled windows: {len(labeled_ds)} | vocab size: {len(id_to_label)}")

    model = IMUToTextModel(
        checkpoint=args.checkpoint,
        qformer_name=args.qformer_name,
        lm_name=args.lm_name,
        device=device_,
    ).to(device_)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device_, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device_, train=False)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "id_to_label": id_to_label,
                    "tokenizer": tokenizer,
                    "qformer_name": args.qformer_name,
                },
                "autoregressor.pt",
            )
            print("Saved improved checkpoint to autoregressor.pt")


if __name__ == "__main__":
    main()
