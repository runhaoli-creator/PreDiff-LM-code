"""
WikiText-103 dataset pipeline for CFlow-LM.

Streams the standard Wikipedia LM benchmark (103M+ tokens) from HuggingFace
datasets, tokenizes with GPT-2 tokenizer, and serves document-aware packed
batches at configurable crop lengths (512/1024/2048).

Reference: Merity et al. (2017) "Pointer Sentinel Mixture Models"
HF dataset: wikitext-103-raw-v1
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class WikiTextDataset(IterableDataset):
    """Document-aware, packed WikiText-103 dataset.

    Args:
        split:         "train" | "validation" | "test".
        tokenizer_name: HF tokenizer id.
        max_length:    Token crop length (512, 1024, or 2048).
        prompt_len:    Number of tokens to use as prompt; rest is target.
        stride:        Token stride between crops (default = max_length → no overlap).
        seed:          Random seed for shuffling.
        streaming:     If True, use HF streaming (memory efficient).
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        prompt_len: int = 256,
        stride: Optional[int] = None,
        seed: int = 42,
        streaming: bool = False,
    ):
        super().__init__()
        self.split = split
        self.max_length = max_length
        self.prompt_len = prompt_len
        self.stride = stride or max_length
        self.seed = seed
        self.streaming = streaming

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split,
            streaming=streaming,
        )

    def _tokenize_document(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __iter__(self) -> Iterator[dict]:
        buffer: list[int] = []
        rng = random.Random(self.seed)

        def iterate_documents():
            if self.streaming:
                for item in self._dataset:
                    yield item["text"]
            else:
                indices = list(range(len(self._dataset)))
                if self.split == "train":
                    rng.shuffle(indices)
                for i in indices:
                    yield self._dataset[i]["text"]

        for text in iterate_documents():
            if not text.strip():
                continue
            tokens = self._tokenize_document(text)
            tokens = tokens + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            # Yield crops from buffer
            while len(buffer) >= self.max_length:
                crop = buffer[: self.max_length]
                buffer = buffer[self.stride:]
                prompt_ids = torch.tensor(crop[: self.prompt_len], dtype=torch.long)
                target_ids = torch.tensor(crop[self.prompt_len :], dtype=torch.long)
                yield {
                    "prompt_ids": prompt_ids,
                    "target_ids": target_ids,
                    "input_ids": torch.tensor(crop, dtype=torch.long),
                }


def build_wikitext_loader(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 512,
    prompt_len: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    streaming: bool = False,
) -> DataLoader:
    """Build a DataLoader for WikiText-103.

    Note: num_workers defaults to 0 because IterableDataset with multiple
    workers duplicates data unless worker splitting logic is implemented.
    """
    dataset = WikiTextDataset(
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
