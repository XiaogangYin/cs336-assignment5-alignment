import os
import random
import json
import itertools

from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
import pathlib

from cs336_alignment.data_helper import *

__all__ = [
    "compute_per_instance_dpo_loss",
]

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    eos = tokenizer.eos_token or "<|end_of_text|>"
    chosen = alpaca_sft_format(prompt, response_chosen, eos=eos)
    rejected = alpaca_sft_format(prompt, response_rejected, eos=eos)
    chosen_ids = tokenizer(chosen,
        return_tensors="pt", return_attention_mask=False).input_ids
    rejected_ids = tokenizer(rejected,
        return_tensors="pt", return_attention_mask=False).input_ids
    #
    def get_logp_ref(model, input_ids):
        # use the loss (ForCausalLMLoss masked cross entropy loss = - logp)
        with torch.no_grad(): 
            result = model(input_ids=input_ids, labels=input_ids)
            # substract 1 , shift one position
            return -result.loss * (input_ids.shape[-1] - 1)

    def get_logp(model, input_ids): 
        result = model(input_ids=input_ids, labels=input_ids)
        return -result.loss * (input_ids.shape[-1] - 1)

    return -F.logsigmoid(beta*(
        get_logp(lm, chosen_ids) - get_logp_ref(lm_ref, chosen_ids)
        - get_logp(lm, rejected_ids) + get_logp_ref(lm_ref, rejected_ids)
        ))


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests/fixtures"

    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2")
    print(type(model))
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = compute_per_instance_dpo_loss(
        lm=model,
        lm_ref=model_ref,
        tokenizer=tokenizer,
        beta=0.5,
        prompt=prompt,
        response_chosen=good_response,
        response_rejected=bad_response,
    )
    print(loss)
