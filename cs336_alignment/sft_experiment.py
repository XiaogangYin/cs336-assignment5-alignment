from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.utils.data import Dataset, DataLoader
from .sft_helper import *

from tqdm import tqdm
import wandb

from dataclasses import dataclass, field

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

#gsm8k format
class JsonlDataset(Dataset):
    def __init__(self, jsonl_file_path):
        examples = []
        with open(input_path) as f:
            for line in f:
                examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]["question"], self.examples[idx]["answer"]

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="./output/qwen2.5-math-sft",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "每设备训练批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "每设备评估批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "预热比例"}
    )
    betas: tuple[float, float] = field(
        default=(0.9, 0.99),
        metadata={"help": "betas"})
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志步数"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "评估步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存步数"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "保存模型数量限制"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "使用FP16混合精度"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "使用BF16混合精度"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "使用梯度检查点"}
    )
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "优化器"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "学习率调度器类型"}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "报告工具"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed配置文件"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "本地rank，用于分布式训练"}
    )


def train():
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_dataset = JsonlDataset("data/gsm8k/test.jsonl")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    gradient_accumulation_steps = 4
    normalize_constant = 10

    optimizer = AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay,
                               betas=config.betas,
                               eps=1e-8)

    for idx, (prompt_strs, output_strs) in tqdm(enumerate(data_loader)):
        train_batch = tokenize_prompt_and_output(prompt_strs, output_strs)
        input_ids = train_batch["input_ids"].to(device)
        labels = train_batch["labels"].to(device)
        response_mask = train_batch["response_mask"].to(device)
        # Forward pass.
        response = get_response_log_probs(model, input_ids, labels, False)
        loss, _ = sft_microbatch_train_step(response["log_probs"],
            response_mask,
            gradient_accumulation_steps,
            normalize_constant)

        if (idx + 1) % gradient_accumulation_steps == 0:
            # Update weights every `gradient_accumulation_steps` batches.
            optimizer.step()
            optimizer.zero_grad()
