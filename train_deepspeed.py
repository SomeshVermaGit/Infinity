"""
DeepSpeed training script for mini-LLM.
Supports ZeRO optimization, mixed precision, and activation checkpointing.
"""
import argparse
import json
import os
import time
from pathlib import Path

import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import MiniLLM, ModelConfig
from tokenizer import Tokenizer


class TextDataset(Dataset):
    """Simple text dataset that returns tokenized sequences."""

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_seq_len: int = 2048):
        """
        Args:
            data_path: Path to preprocessed token file (each line = space-separated token IDs)
                       or text file (will be tokenized on the fly)
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Try to parse as pre-tokenized IDs
                try:
                    ids = [int(x) for x in line.split()]
                except ValueError:
                    # Otherwise tokenize the text
                    ids = tokenizer.encode(line, add_bos=True, add_eos=True)

                # Chunk into max_seq_len segments
                for i in range(0, len(ids), max_seq_len):
                    chunk = ids[i : i + max_seq_len]
                    if len(chunk) > 1:  # Need at least 2 tokens for autoregressive training
                        self.samples.append(chunk)

        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Pad to max_seq_len
        if len(ids) < self.max_seq_len:
            ids = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def train(args):
    # Load tokenizer
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # Model config
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_flash_attention=args.use_flash_attention,
    )

    # Create model
    model = MiniLLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    # Dataset and dataloader
    dataset = TextDataset(args.data_path, tokenizer, max_seq_len=args.max_seq_len)
    # Note: DeepSpeed will handle the batch size internally; this is the per-GPU micro batch
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model_engine.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            # Forward pass
            logits, loss = model_engine(input_ids, labels=labels)

            # Backward and optimizer step (DeepSpeed handles this)
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": optimizer.param_groups[0]["lr"]})

            # Save checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(model_engine, tokenizer, config, args.output_dir, global_step)

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} completed. Avg loss: {avg_epoch_loss:.4f}, Time: {elapsed:.2f}s")

        # Save epoch checkpoint
        save_checkpoint(model_engine, tokenizer, config, args.output_dir, f"epoch_{epoch + 1}")

    print(f"Training completed in {time.time() - start_time:.2f}s")


def save_checkpoint(model_engine, tokenizer, config, output_dir, tag):
    """Save DeepSpeed checkpoint with metadata."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save DeepSpeed checkpoint (handles ZeRO state)
    model_engine.save_checkpoint(checkpoint_dir, tag=str(tag))

    # Save config
    config_dict = {
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_flash_attention", action="store_true", help="Use flash attention if available")

    # Data args
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer model")

    # Training args
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    # DeepSpeed args (loaded from config file)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")

    # Parse and add DeepSpeed config
    args = parser.parse_args()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Override with DeepSpeed config file
    args.deepspeed_config = args.deepspeed_config

    train(args)


if __name__ == "__main__":
    main()
