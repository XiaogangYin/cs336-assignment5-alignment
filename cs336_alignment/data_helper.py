import os
import random
import json
import itertools
from typing import Any, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


__all__ = [
    "alpaca_sft_format",
    "PackedSftDataset",
    "iterate_batches",
]

with open("cs336_alignment/prompts/alpaca_sft.prompt") as f:
    alpaca_sft_prompt = f.read().rstrip()

def alpaca_sft_format(prompt, response):
    return alpaca_sft_prompt.replace("{instruction}", prompt
                    ).replace("{response}", response
                    ) + "<|end_of_text|>"


class PackedSftDataset(Dataset):
    """
    Implement a PyTorch Dataset subclass that generates examples for instruction
    tuning. The Dataset should have the following interface
    """
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        """
        Constructs the dataset.
        tokenizer is a transformers tokenizer for use in tokenizing and encoding the instruction
        tuning data. 
        dataset_path is a path to instruction tuning data. 
        seq_length is the desired
        length of sequences to generate from the dataset (typically the desired language model context
        length). 
        shuffle controls whether or not documents are shuffled before concatenation
        (when shuffle=True), or if they are concatenated in the order they appear in the data
        (when shuffle=False).
        """
        DELIMITER_ID = 128001

        with open("cs336_alignment/prompts/alpaca_sft.prompt") as f:
            alpaca_sft_prompt = f.read().rstrip()

        examples = []
        with open(dataset_path) as f:
            for line in f:
                example = json.loads(line)
                example_str = alpaca_sft_prompt.replace(
                        "{instruction}", example["prompt"]
                    ).replace(
                        "{response}", example["response"]
                    ) + "<|end_of_text|>"
                encoded = tokenizer(example_str, 
                        return_tensors="pt", return_attention_mask=False)
                examples.append(encoded["input_ids"][0])
        if shuffle:
            random.shuffle(examples)

        self.encoded_ids = torch.cat(examples, dim=0)
        self.seq_length = seq_length

    def __len__(self):
        return self.encoded_ids.shape[0] // self.seq_length

    def __getitem__(self, i):
        # must have this, otherwise loop foreever for example in  packed_sft_dataset
        if i < 0 or i >= len(self):
            raise IndexError("Index out of range")

        s = i * self.seq_length
        return {
            "input_ids": self.encoded_ids[s : s+self.seq_length],
            "labels": self.encoded_ids[s+1 : s+self.seq_length+1],
        } 

def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import pathlib

    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"


    sft_sample_path = FIXTURES_PATH / "sft_sample.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / "Meta-Llama-3-8B")
    seq_length = 32
    packed_sft_dataset = PackedSftDataset(
        tokenizer=tokenizer,
        dataset_path=sft_sample_path,
        seq_length=seq_length,
        shuffle=False,
    )
    print(len(packed_sft_dataset))
    for i, example in  enumerate(packed_sft_dataset):
        print(i, example["input_ids"].shape)
        