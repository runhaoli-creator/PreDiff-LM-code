"""
PG-19 dataset pipeline for CFlow-LM.

Streams the Project Gutenberg 19th-century books corpus from HuggingFace
datasets, used as a long-context generation benchmark. Books are tokenized
and packed into document-aware crops.

Reference: Rae et al. (2020) "Compressive Transformers for Long-Range Sequence Modelling"
HF dataset: pg19
"""

from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class PG19Dataset(IterableDataset):
    """Document-aware PG-19 long-book dataset.

    PG-19 books are very long (100K+ tokens each), so we crop random
    windows during training and sliding windows for evaluation.

    Args:
        split:          "train" | "validation" | "test".
        tokenizer_name: HF tokenizer id.
        max_length:     Token crop length.
        prompt_len:     Number of tokens used as prompt.
        seed:           Random seed.
        streaming:      Use HF streaming mode.
        crops_per_doc:  Max crops to take from each book (None = all).
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
        prompt_len: int = 512,
        seed: int = 42,
        streaming: bool = False,
        crops_per_doc: Optional[int] = None,
    ):
        super().__init__()
        self.split = split
        self.max_length = max_length
        self.prompt_len = prompt_len
        self.seed = seed
        self.streaming = streaming
        self.crops_per_doc = crops_per_doc

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._dataset = load_dataset(
            "emozilla/pg19",
            split=split,
            streaming=streaming,
        )

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)

        def iterate_books():
            if self.streaming:
                for item in self._dataset:
                    yield item["text"]
            else:
                indices = list(range(len(self._dataset)))
                if self.split == "train":
                    rng.shuffle(indices)
                for i in indices:
                    yield self._dataset[i]["text"]

        for text in iterate_books():
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens = tokens + [self.tokenizer.eos_token_id]

            if len(tokens) < self.max_length:
                continue

            # Extract crops
            crop_count = 0
            if self.split == "train":
                # Random crops
                max_start = len(tokens) - self.max_length
                num_crops = self.crops_per_doc or max(1, max_start // self.max_length)
                for _ in range(num_crops):
                    start = rng.randint(0, max_start)
                    crop = tokens[start : start + self.max_length]
                    prompt_ids = torch.tensor(crop[: self.prompt_len], dtype=torch.long)
                    target_ids = torch.tensor(crop[self.prompt_len :], dtype=torch.long)
                    yield {
                        "prompt_ids": prompt_ids,
                        "target_ids": target_ids,
                        "input_ids": torch.tensor(crop, dtype=torch.long),
                    }
                    crop_count += 1
                    if self.crops_per_doc and crop_count >= self.crops_per_doc:
                        break
            else:
                # Sliding window for eval
                for start in range(0, len(tokens) - self.max_length + 1, self.max_length):
                    crop = tokens[start : start + self.max_length]
                    prompt_ids = torch.tensor(crop[: self.prompt_len], dtype=torch.long)
                    target_ids = torch.tensor(crop[self.prompt_len :], dtype=torch.long)
                    yield {
                        "prompt_ids": prompt_ids,
                        "target_ids": target_ids,
                        "input_ids": torch.tensor(crop, dtype=torch.long),
                    }


def build_pg19_loader(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    prompt_len: int = 512,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 42,
    streaming: bool = False,
    crops_per_doc: Optional[int] = None,
) -> DataLoader:
    """Build a DataLoader for PG-19.

    Note: num_workers defaults to 0 because IterableDataset with multiple
    workers duplicates data unless worker splitting logic is implemented.
    """
    dataset = PG19Dataset(
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        prompt_len=prompt_len,
        seed=seed,
        streaming=streaming,
        crops_per_doc=crops_per_doc,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
