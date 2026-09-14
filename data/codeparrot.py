"""
CodeParrot-clean dataset pipeline for CFlow-LM.

Deduplicated Python code files from GitHub. Used for structured code
generation evaluation (exact match, pass@1-style metrics).

Reference: von Werra et al. (2021) "CodeParrot: Training a Code Generation Model from Scratch"
HF dataset: codeparrot/codeparrot-clean
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Default split sizes (approximate)
_CODEPARROT_TRAIN_SIZE = 5_000_000
_CODEPARROT_VAL_SIZE   =     5_000


class CodeParrotDataset(IterableDataset):
    """Document-aware CodeParrot-clean Python code dataset.

    Each document is a single Python file. We tokenize the file and serve
    packed crops, splitting each crop into prompt (context) and target
    (code continuation to generate).

    Args:
        split:          "train" | "valid".
        tokenizer_name: HF tokenizer id (use gpt2 for fair comparison).
        max_length:     Sequence crop length.
        prompt_len:     Tokens used as prompt / context.
        seed:           Random seed.
        streaming:      Use HF streaming mode.
        max_file_len:   Skip files longer than this (avoids very large files).
        pack_documents: If True, pack multiple short files into one sequence.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        prompt_len: int = 256,
        seed: int = 42,
        streaming: bool = True,
        max_file_len: int = 50_000,
        pack_documents: bool = True,
    ):
        super().__init__()
        self.split = split
        self.max_length = max_length
        self.prompt_len = prompt_len
        self.seed = seed
        self.max_file_len = max_file_len
        self.pack_documents = pack_documents

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Map "valid" to the HF split name
        hf_split = "valid" if split == "valid" else split
        self._dataset = load_dataset(
            "codeparrot/codeparrot-clean",
            split=hf_split,
            streaming=streaming,
        )

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)
        buffer: list[int] = []

        eos_id = self.tokenizer.eos_token_id

        for item in self._dataset:
            code = item.get("content", "")
            if not code.strip():
                continue
            if len(code) > self.max_file_len:
                code = code[: self.max_file_len]

            tokens = self.tokenizer.encode(code, add_special_tokens=False)
            tokens = tokens + [eos_id]  # file separator

            if self.pack_documents:
                buffer.extend(tokens)
                while len(buffer) >= self.max_length:
                    crop = buffer[: self.max_length]
                    buffer = buffer[self.max_length:]
                    yield self._make_example(crop)
            else:
                # Each file is its own example (pad/truncate)
                if len(tokens) < self.max_length:
                    # pad right
                    tokens = tokens + [eos_id] * (self.max_length - len(tokens))
                else:
                    tokens = tokens[: self.max_length]
                yield self._make_example(tokens)

    def _make_example(self, crop: list[int]) -> dict:
        prompt_ids = torch.tensor(crop[: self.prompt_len], dtype=torch.long)
        target_ids = torch.tensor(crop[self.prompt_len :], dtype=torch.long)
        return {
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "input_ids": torch.tensor(crop, dtype=torch.long),
        }


def build_codeparrot_loader(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    prompt_len: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    streaming: bool = True,
) -> DataLoader:
    """Build a DataLoader for CodeParrot-clean.

    Note: num_workers defaults to 0 because IterableDataset with multiple
    workers duplicates data unless worker splitting logic is implemented.
    """
    dataset = CodeParrotDataset(
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        prompt_len=prompt_len,
        seed=seed,
        streaming=streaming,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
