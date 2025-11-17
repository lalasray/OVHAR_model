from __future__ import annotations

from pathlib import Path
from typing import Tuple

import sys
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Blip2QFormerConfig,
    Blip2QFormerModel,
)

ROOT = Path(__file__).resolve().parents[2]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from model.Quantizer.imu_OVHAR import VQVAEIMU


class FrozenVQVAEEncoder(nn.Module):
    """Load a trained VQVAEIMU and expose its encoder/pre_vq without gradients."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        args = ckpt.get("args", {})
        self.model = VQVAEIMU(
            num_hiddens=args.get("num_hiddens", 128),
            num_residual_layers=args.get("num_residual_layers", 2),
            num_residual_hiddens=args.get("num_residual_hiddens", 32),
            num_embeddings=args.get("num_embeddings", 512),
            embedding_dim=args.get("embedding_dim", 64),
            commitment_cost=args.get("commitment_cost", 0.25),
            decay=args.get("decay", 0.99),
        )
        self.model.load_state_dict(ckpt["model_state"])
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.model._encoder(x)
            z = self.model._pre_vq_conv(z)
            # shape: [B, D, T]
        return z


class HFQFormer(nn.Module):
    """HuggingFace BLIP-2 Q-Former block with learnable queries projected from encoder feats."""

    def __init__(self, name: str, input_dim: int):
        super().__init__()
        self.config = Blip2QFormerConfig.from_pretrained(name)
        self.qformer = Blip2QFormerModel.from_pretrained(name, config=self.config)
        self.query_tokens = nn.Parameter(torch.randn(self.config.num_query_tokens, self.config.hidden_size))
        self.proj = nn.Linear(input_dim, self.config.hidden_size)

    def forward(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        # encoder_feats: [B, T, D]
        B, T, _ = encoder_feats.shape
        queries = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, H]
        encoder_hidden = self.proj(encoder_feats)  # [B, T, H]
        attention_mask = torch.ones((B, T), device=encoder_feats.device, dtype=torch.long)
        outputs = self.qformer(
            query_embeds=queries,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # [B, Q, H]


class FrozenLLM(nn.Module):
    """Frozen causal LM plus tokenizer to generate/score label text."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B", device: torch.device | str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(
        self,
        prefix_embed: torch.Tensor,
        label_input_ids: torch.Tensor,
        label_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prefix_embed: [B, 1, H] projected Q-Former token in LLM hidden space.
        label_input_ids: [B, L] tokenized label text.
        label_attention_mask: [B, L]
        """
        lm = self.model
        tok_embeds = lm.transformer.wte(label_input_ids)
        inputs_embeds = torch.cat([prefix_embed, tok_embeds], dim=1)  # [B, 1+L, H]

        # Labels: ignore loss on prefix token
        prefix_labels = torch.full((label_input_ids.size(0), 1), -100, device=label_input_ids.device)
        labels = torch.cat([prefix_labels, label_input_ids], dim=1)
        attn_mask = torch.cat(
            [torch.ones((label_input_ids.size(0), 1), device=label_attention_mask.device), label_attention_mask], dim=1
        )

        outputs = lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        # outputs.loss is per-token averaged
        return outputs.loss, outputs.logits


class IMUToTextModel(nn.Module):
    def __init__(
        self,
        checkpoint: Path,
        qformer_name: str = "Salesforce/blip2-opt-2.7b",
        lm_name: str = "meta-llama/Meta-Llama-3.1-8B",
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.encoder = FrozenVQVAEEncoder(checkpoint)
        self.qformer = HFQFormer(name=qformer_name, input_dim=self.encoder.model._pre_vq_conv.out_channels)
        self.proj = nn.Conv1d(
            self.encoder.model._pre_vq_conv.out_channels, self.qformer.config.hidden_size, kernel_size=1
        )
        self.lm = FrozenLLM(model_name=lm_name, device=device)
        self.q_to_lm = nn.Linear(self.qformer.config.hidden_size, self.lm.hidden_size)

    def forward(
        self, x: torch.Tensor, label_input_ids: torch.Tensor, label_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = self.encoder(x)  # [B, D, T]
        z_proj = self.proj(z).transpose(1, 2)  # [B, T, dim]
        q_tokens = self.qformer(z_proj)  # [B, Q, dim]
        pooled = q_tokens.mean(dim=1).unsqueeze(1)  # [B,1,dim]
        prefix = self.q_to_lm(pooled)  # [B,1,H]
        loss, logits = self.lm(prefix, label_input_ids, label_attention_mask)
        return loss, logits
