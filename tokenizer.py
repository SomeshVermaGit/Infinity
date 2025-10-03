"""
Tokenizer wrapper for SentencePiece with training and encoding utilities.
"""
import os
from typing import List, Optional
import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tokenizer. If model_path is provided, load existing model.

        Args:
            model_path: Path to trained .model file
        """
        self.sp = None
        if model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)

    def train(
        self,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        """
        Train a SentencePiece tokenizer.

        Args:
            input_file: Path to training corpus (text file, one sentence per line)
            model_prefix: Output model name (will create {model_prefix}.model and .vocab)
            vocab_size: Vocabulary size
            model_type: "bpe" or "unigram"
            character_coverage: Character coverage (1.0 for languages like English)
            pad_id, unk_id, bos_id, eos_id: Special token IDs
        """
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
            num_threads=os.cpu_count(),
        )
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs."""
        assert self.sp is not None, "Tokenizer not trained or loaded"
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        assert self.sp is not None, "Tokenizer not trained or loaded"
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        assert self.sp is not None, "Tokenizer not trained or loaded"
        return self.sp.vocab_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()


if __name__ == "__main__":
    # Example: train a tokenizer on a sample corpus
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--prefix", type=str, default="tokenizer", help="Model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocab size")
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe", "unigram"])
    args = parser.parse_args()

    tokenizer = Tokenizer()
    tokenizer.train(
        input_file=args.input,
        model_prefix=args.prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )

    # Test
    test_text = "Hello, world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")
