import argparse
import json
import logging
import sys
from typing import Callable,List
from statistics import mean

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from xopen import xopen

logger = logging.getLogger(__name__)

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[int],    
    eval_sampling_params: SamplingParams,
    output_path: str,
    input_examples: List[str],
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    assert len(outputs) == len(prompts)
    logger.info(f"Processed {len(prompts)} prompts")

    # Print the outputs.
    rewards = []
    generated_texts = []
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        reward = reward_fn(generated_text, ground_truth)
        rewards.append(reward)

    all_metrics = []
    with xopen(output_path, "w") as fout:
        for input_example, ground_truth, response, prompt, reward in tqdm(
            zip(input_examples, ground_truths, generated_texts, prompts, rewards)
        ):
            all_metrics.append(reward)

            fout.write(
                json.dumps(
                    {
                        "model_prompt": prompt,
                        "input_example": input_example,
                        "model_response": response,
                        "ground_truth": ground_truth,
                        "reward": reward,
                    }
                )
                + "\n"
            )

    for key in sorted(list(all_metrics[0].keys())):
        metric_value = mean([metrics[key] for metrics in all_metrics])
        logger.info(f"{key}: {metric_value}")

def main(input_path, model_name_or_path, num_gpus=1, output_path="tmp/tmp.txt"):
    input_examples = []
    with xopen(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    logger.info(f"Read {len(input_examples)} model responses from {input_path}")

    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        r1_zero_prompt = f.read()

    prompts = []
    ground_truths = []
    for example in input_examples:
        ground_truths.append(int(example["answer"].split("####")[-1].replace(",", "")))

        prompts.append(r1_zero_prompt.replace("{question}", example["question"]))

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
    )

    evaluate_vllm(model,  r1_zero_reward_fn, prompts,
         ground_truths, sampling_params, output_path, input_examples)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path", 
        help="HF name of the model to use",
        required=True,
        default="Qwen/Qwen2.5-Math-1.5B",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model_name_or_path,
        args.num_gpus,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
'''
cmd:
uv run python cs336_alignment/eval_math_baseline.py --input-path data/gsm8k/test.jsonl \
 --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
 --output-path gsm8k.test.qwen2.5.math.1.5B.result.jsonl 1>eval_math_baseline.log 2>&1
 '''