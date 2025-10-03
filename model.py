"""
From-scratch Transformer implementation with causal attention.
Optimized for DeepSpeed ZeRO + activation checkpointing.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_seq_len: int = 2048
    dropout: float = 0.1
    bias: bool = True
    use_flash_attention: bool = False  # Set True if flash_attn is available


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create upper-triangular causal mask (True = masked positions)."""
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


class ScaledDotProductAttention(nn.Module):
    """Vanilla scaled dot-product attention with causal masking."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k, v: (B, num_heads, T, head_dim)
            mask: (T, T) boolean mask (True = masked)

        Returns:
            out: (B, num_heads, T, head_dim)
            attn: (B, num_heads, T, T) attention weights
        """
        d_k = q.size(-1)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d_k)  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.embed_dim = config.embed_dim
        self.use_flash = config.use_flash_attention

        # Single projection for Q, K, V
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn = ScaledDotProductAttention(config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, E)
            mask: (T, T) causal mask

        Returns:
            out: (B, T, E)
        """
        B, T, E = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3E)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, d)

        if self.use_flash:
            # Use PyTorch 2.0 flash attention (recommended)
            # Note: requires PyTorch >= 2.0
            try:
                # Reshape for F.scaled_dot_product_attention
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
                )
            except Exception:
                # Fallback to manual attention
                out, _ = self.attn(q, k, v, mask=mask)
        else:
            out, _ = self.attn(q, k, v, mask=mask)

        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, E)  # (B, T, E)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network (MLP)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.embed_dim
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block: LayerNorm -> Attention -> LayerNorm -> FFN with residuals."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class MiniLLM(nn.Module):
    """Mini autoregressive Transformer LLM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Output head (tied with token embedding if desired)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (optional but recommended)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token indices
            labels: (B, T) target token indices (for training)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss (if labels provided)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        tok_emb = self.token_embedding(input_ids)  # (B, T, E)
        pos_emb = self.position_embedding(positions)  # (1, T, E)
        x = self.dropout(tok_emb + pos_emb)

        # Causal mask
        mask = causal_mask(T, device)

        # Transformer blocks (wrap in checkpointing for memory efficiency)
        for block in self.blocks:
            # For activation checkpointing, wrap block in torch.utils.checkpoint.checkpoint
            # or use DeepSpeed's activation checkpointing API
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Standard cross-entropy loss (shift logits/labels for autoregressive)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation (greedy or sampling).

        Args:
            input_ids: (B, T) prompt tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, sample from top-k tokens

        Returns:
            generated: (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_ids_cond = (
                input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len :]
            )
            logits, _ = self.forward(input_ids_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    # Quick test
    config = ModelConfig(vocab_size=1000, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=512)
    model = MiniLLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    logits, loss = model(input_ids, labels=labels)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    # Generation test
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
