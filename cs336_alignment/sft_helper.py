

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from torch.nn.functional import softmax

__all__ = [
    "tokenize_prompt_and_output",
    "compute_entropy",
    "get_response_log_probs",
]

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """    
    encoded = tokenizer(text=prompt_strs,
                text_pair=output_strs, 
                return_token_type_ids=True,
                return_attention_mask=False, 
                return_tensors="pt",
                padding=True)
    return {
        "input_ids": encoded['input_ids'][:, :-1],
        "labels": encoded['input_ids'][:, 1:],
        "response_mask": encoded['token_type_ids'][:, 1:],
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    logits -= torch.max(logits)
    exp_x = torch.exp(logits)
    p = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    entropy = - torch.sum(p * torch.log(p), dim=-1)
    return entropy

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    targets_x = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {"log_probs": targets_x - torch.logsumexp(logits, dim=-1)}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
