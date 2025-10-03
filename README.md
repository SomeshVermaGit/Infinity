# Mini-LLM: From-Scratch Transformer with DeepSpeed

A minimal, production-ready implementation of a custom Transformer LLM trained at scale with PyTorch + DeepSpeed. Features from-scratch attention, ZeRO optimization, mixed precision, and activation checkpointing.

## Features

- **From-scratch Transformer**: Custom multi-head causal attention, feed-forward networks, and layer normalization
- **DeepSpeed Integration**: ZeRO stage 2/3, optimizer offloading, gradient accumulation
- **Memory Optimizations**: Activation checkpointing, mixed precision (FP16/BF16), optional FlashAttention
- **SentencePiece Tokenizer**: BPE/Unigram tokenization with deterministic vocab
- **Distributed Training**: Single-node multi-GPU or multi-node support

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install flash-attention for faster training (requires CUDA)
# pip install flash-attn --no-build-isolation
```

### 2. Prepare Data

Create a training corpus (one sentence per line):

```bash
# Example: create sample data
cat > sample_data.txt << EOF
The quick brown fox jumps over the lazy dog.
Machine learning is transforming the world.
Deep learning models require large datasets.
Transformers use self-attention mechanisms.
EOF
```

### 3. Train Tokenizer

```bash
python tokenizer.py \
  --input sample_data.txt \
  --prefix tokenizer \
  --vocab_size 32000 \
  --model_type bpe
```

This creates `tokenizer.model` and `tokenizer.vocab`.

### 4. Single-GPU Training (Test Run)

First, test on a single GPU without DeepSpeed:

```bash
python -c "
from model import MiniLLM, ModelConfig
import torch

config = ModelConfig(vocab_size=32000, embed_dim=256, num_heads=8, num_layers=6)
model = MiniLLM(config)
print(f'Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

# Test forward pass
x = torch.randint(0, 32000, (2, 64))
logits, loss = model(x, labels=x)
print(f'Loss: {loss.item():.4f}')
"
```

### 5. Multi-GPU Training with DeepSpeed

Launch distributed training:

```bash
# Single node with 4 GPUs
deepspeed --num_gpus=4 train_deepspeed.py \
  --data_path sample_data.txt \
  --tokenizer_path tokenizer.model \
  --embed_dim 768 \
  --num_heads 12 \
  --num_layers 12 \
  --max_seq_len 2048 \
  --epochs 3 \
  --micro_batch_size 4 \
  --output_dir ./checkpoints \
  --deepspeed_config ds_config.json

# Multi-node (e.g., 2 nodes with 8 GPUs each)
# On each node, run:
deepspeed --num_nodes=2 --num_gpus=8 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  train_deepspeed.py \
  --data_path sample_data.txt \
  --tokenizer_path tokenizer.model \
  --embed_dim 768 \
  --num_heads 12 \
  --num_layers 12 \
  --deepspeed_config ds_config.json
```

## Configuration

### Model Sizes

| Model | Embed Dim | Heads | Layers | Params | Config |
|-------|-----------|-------|--------|--------|--------|
| Tiny  | 256       | 8     | 6      | ~10M   | Single GPU |
| Small | 512       | 8     | 8      | ~40M   | Single GPU + ZeRO |
| Medium| 768       | 12    | 12     | ~100M  | Multi-GPU + ZeRO stage 2 |
| Large | 1024      | 16    | 24     | ~350M  | Multi-GPU + ZeRO stage 3 |

### DeepSpeed Config (`ds_config.json`)

Key parameters to tune:

- **`train_batch_size`**: Global batch size (should equal `micro_batch_size * num_gpus * gradient_accumulation_steps`)
- **`train_micro_batch_size_per_gpu`**: Per-GPU batch size
- **`gradient_accumulation_steps`**: Number of steps to accumulate gradients
- **ZeRO stage**:
  - Stage 1: Partition optimizer states
  - Stage 2: Partition optimizer states + gradients (recommended)
  - Stage 3: Partition optimizer states + gradients + parameters (max memory savings)
- **`fp16` vs `bf16`**: Use BF16 on A100/H100, otherwise FP16
- **`activation_checkpointing`**: Enable for large models (trades compute for memory)

Example for max memory efficiency (1B+ params):

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "bf16": {"enabled": true},
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

## Memory Optimization Checklist

If you encounter OOM errors, try these in order:

1. **Enable mixed precision**: Set `"fp16": {"enabled": true}` or `"bf16": {"enabled": true}`
2. **Reduce micro batch size**: Lower `train_micro_batch_size_per_gpu` (e.g., from 4 to 2 or 1)
3. **Enable activation checkpointing**: Already enabled in default config
4. **Upgrade ZeRO stage**: Change `"stage": 2` to `"stage": 3`
5. **Offload to CPU**: Enable `"offload_param": {"device": "cpu"}` in ZeRO config
6. **Use FlashAttention**: Set `--use_flash_attention` and install `flash-attn`
7. **Reduce sequence length**: Lower `--max_seq_len` (e.g., from 2048 to 1024)

## Monitoring & Debugging

### Check GPU Memory

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

### Profile with PyTorch

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    logits, loss = model(input_ids, labels=labels)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### DeepSpeed Logging

Enable detailed logging in `ds_config.json`:

```json
{
  "steps_per_print": 1,
  "wall_clock_breakdown": true,
  "dump_state": true
}
```

## Generation (Inference)

```python
from model import MiniLLM, ModelConfig
from tokenizer import Tokenizer
import torch

# Load tokenizer and model
tokenizer = Tokenizer("tokenizer.model")
config = ModelConfig(vocab_size=tokenizer.vocab_size)
model = MiniLLM(config)

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint-1000/model.pt")
model.load_state_dict(checkpoint)
model.eval()

# Generate
prompt = "The future of AI is"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
generated = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
output = tokenizer.decode(generated[0].tolist())
print(output)
```

## Project Structure

```
mini-llm/
├── model.py              # Transformer implementation
├── tokenizer.py          # SentencePiece wrapper
├── train_deepspeed.py    # Training script
├── ds_config.json        # DeepSpeed configuration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Advanced: FlashAttention

To use FlashAttention for 2-3x faster training and 5-10x memory reduction:

1. Install: `pip install flash-attn --no-build-isolation`
2. Add `--use_flash_attention` flag to training command
3. Or set `use_flash_attention=True` in ModelConfig

The code will automatically fall back to PyTorch's built-in `scaled_dot_product_attention` if FlashAttention is not available.

## Tips

- **Start small**: Test single-GPU with tiny model first, then scale up
- **Monitor loss spikes**: Check for numerical instability with FP16 (switch to BF16 if needed)
- **Deterministic training**: Set seeds for reproducibility:
  ```python
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  ```
- **Resume training**: Use `model_engine.load_checkpoint()` to resume from DeepSpeed checkpoints
- **Hyperparameter tuning**: Start with lr=5e-5, warmup=1000 steps, adjust based on loss curves

## References

- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [DeepSpeed Activation Checkpointing](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## License

MIT

---

**Ready to scale?** Start with the quick start guide above, then adjust batch sizes and model dimensions for your hardware. For production training, use multi-node setup with ZeRO stage 3 + CPU offload.
#   I n f i n i t y  
 