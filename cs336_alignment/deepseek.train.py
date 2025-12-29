以下是使用 Qwen2.5-Math-1.5B 和数学数据集进行 SFT（监督微调）的完整训练代码。我将提供一个端到端的解决方案，包括数据处理、训练和评估。

1. 环境配置
bash
# 创建环境
pip install torch transformers datasets peft accelerate bitsandbytes
pip install sentencepiece protobuf scipy sklearn tqdm matplotlib
pip install wandb  # 可选，用于实验追踪
2. 完整的训练脚本
python
#!/usr/bin/env python3
"""
Qwen2.5-Math-1.5B SFT 训练脚本
支持 LoRA/QLoRA 高效微调
"""

import os
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict
import json
from tqdm import tqdm
import numpy as np

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import wandb

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-Math-1.5B",
        metadata={"help": "模型路径或名称"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用LoRA"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA目标模块，逗号分隔"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "使用4位量化"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "4位量化的计算数据类型"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4位量化类型"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="math",
        metadata={"help": "数据集名称"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "数据集配置名称"}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "训练数据文件"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据文件"}
    )
    max_length: int = field(
        default=1024,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "数据预处理工作线程数"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "覆盖缓存"}
    )

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

def load_math_datasets(data_args):
    """加载数学数据集"""
    
    # 这里可以加载多个数学数据集
    datasets = []
    
    # 1. GSM8K数据集
    print("加载GSM8K数据集...")
    try:
        gsm8k = load_dataset("gsm8k", "main")
        # 格式化GSM8K数据
        def format_gsm8k(example):
            return {
                "instruction": "Solve the following math problem step by step.",
                "input": example["question"],
                "output": example["answer"]
            }
        gsm8k = gsm8k.map(format_gsm8k)
        datasets.append(gsm8k)
    except Exception as e:
        print(f"加载GSM8K失败: {e}")
    
    # 2. MATH数据集
    print("加载MATH数据集...")
    try:
        math_dataset = load_dataset("competition_math")
        def format_math(example):
            return {
                "instruction": "Solve the following math competition problem.",
                "input": example["problem"],
                "output": example["solution"]
            }
        math_dataset = math_dataset.map(format_math)
        datasets.append(math_dataset)
    except Exception as e:
        print(f"加载MATH数据集失败: {e}")
    
    # 3. 如果有自定义数据文件
    if data_args.train_file and os.path.exists(data_args.train_file):
        print(f"加载自定义数据: {data_args.train_file}")
        custom_data = load_dataset('json', data_files=data_args.train_file)
        datasets.append(custom_data)
    
    if len(datasets) == 0:
        raise ValueError("没有加载到任何数据集！")
    
    # 合并数据集
    combined_dataset = concatenate_datasets([d['train'] for d in datasets])
    
    # 划分训练/验证集
    dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset

def preprocess_function(examples, tokenizer, data_args):
    """预处理函数"""
    
    # 构建对话格式
    def build_qwen_chat_format(instruction, input_text, output_text):
        # Qwen2.5的对话格式
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
            {"role": "assistant", "content": output_text}
        ]
        
        # 转换为Qwen格式
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    
    inputs = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        
        text = build_qwen_chat_format(instruction, input_text, output_text)
        inputs.append(text)
    
    # 分词
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_length,
        truncation=True,
        padding=False
    )
    
    # 创建标签（对于Causal LM，标签就是输入本身）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs

def train():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", type=str, help="模型参数JSON")
    parser.add_argument("--data_args", type=str, help="数据参数JSON")
    parser.add_argument("--training_args", type=str, help="训练参数JSON")
    
    args, _ = parser.parse_known_args()
    
    # 从JSON加载参数或使用默认值
    if args.model_args:
        with open(args.model_args, 'r') as f:
            model_args_dict = json.load(f)
        model_args = ModelArguments(**model_args_dict)
    else:
        model_args = ModelArguments()
    
    if args.data_args:
        with open(args.data_args, 'r') as f:
            data_args_dict = json.load(f)
        data_args = DataTrainingArguments(**data_args_dict)
    else:
        data_args = DataTrainingArguments()
    
    if args.training_args:
        with open(args.training_args, 'r') as f:
            training_args_dict = json.load(f)
        training_args = TrainingArguments(**training_args_dict)
    else:
        training_args = TrainingArguments()
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 初始化wandb
    if training_args.report_to == "wandb":
        wandb.init(
            project="qwen-math-sft",
            name=f"qwen2.5-math-{training_args.seed}",
            config={
                **model_args.__dict__,
                **data_args.__dict__,
                **training_args.__dict__
            }
        )
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    print("加载和预处理数据集...")
    dataset = load_math_datasets(data_args)
    
    # 预处理数据集
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, data_args),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="预处理数据集"
    )
    
    # 加载模型
    print("加载模型...")
    
    bnb_config = None
    if model_args.use_4bit:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32
    )
    
    # 准备k-bit训练
    if model_args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    if model_args.use_lora:
        print("应用LoRA配置...")
        lora_target_modules = model_args.lora_target_modules.split(",")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=lora_target_modules,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        eval_strategy="steps",
        eval_steps=training_args.eval_steps,
        save_strategy="steps",
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training_args.optim,
        lr_scheduler_type=training_args.lr_scheduler_type,
        report_to=training_args.report_to if training_args.report_to != "none" else None,
        seed=training_args.seed,
        data_seed=training_args.seed,
        ddp_find_unused_parameters=False,
        deepspeed=training_args.deepspeed,
        local_rank=training_args.local_rank,
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=data_args.preprocessing_num_workers,
    )
    
    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print("保存最终模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # 保存LoRA适配器（如果使用）
    if model_args.use_lora:
        model.save_pretrained(os.path.join(training_args.output_dir, "lora_adapter"))
    
    print(f"训练完成！模型保存在: {training_args.output_dir}")

if __name__ == "__main__":
    train()
3. 配置文件示例
创建 config/model_args.json:

json
{
    "model_name_or_path": "Qwen/Qwen2.5-Math-1.5B",
    "use_lora": true,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "lora_target_modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
    "use_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4"
}
创建 config/training_args.json:

json
{
    "output_dir": "./output/qwen2.5-math-sft",
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 500,
    "save_total_limit": 3,
    "fp16": true,
    "bf16": false,
    "gradient_checkpointing": true,
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "report_to": "wandb",
    "seed": 42
}
4. 启动训练脚本
bash
# 基本训练
python train_sft.py \
    --model_args config/model_args.json \
    --training_args config/training_args.json

# 使用DeepSpeed（内存优化）
python train_sft.py \
    --model_args config/model_args.json \
    --training_args config/training_args.json \
    --deepspeed ds_config.json
5. DeepSpeed 配置文件 (ds_config.json)
json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "gradient_clipping": "auto",
    "wall_clock_breakdown": false
}
6. 评估脚本
python
#!/usr/bin/env python3
"""
评估微调后的模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import json
from tqdm import tqdm

def evaluate_model():
    model_path = "./output/qwen2.5-math-sft"
    
    # 加载基础模型
    print("加载基础模型...")
    base_model = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # 加载微调后的模型（如果是LoRA，需要合并）
    if "lora_adapter" in model_path:
        # 加载LoRA适配器
        config = PeftConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # 加载全量微调的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.eval()
    
    # 准备测试数据
    test_dataset = load_dataset("gsm8k", "main", split="test")
    
    results = []
    correct = 0
    total = 0
    
    for i, example in enumerate(tqdm(test_dataset[:100])):  # 测试前100个
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {example['question']}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的答案
        generated_answer = response.split("Answer:")[-1].strip()
        
        # 简单评估（实际评估可能需要更复杂的逻辑）
        # 这里只是示例，实际数学评估需要提取最终数值答案
        results.append({
            "question": example["question"],
            "generated": generated_answer,
            "reference": example["answer"]
        })
        
        # 简单的字符串匹配评估
        if any(str(num) in generated_answer for num in range(100)):
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
    # 保存结果
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印一些示例
    print("\n=== 示例输出 ===")
    for i in range(min(3, len(results))):
        print(f"\n问题: {results[i]['question']}")
        print(f"生成: {results[i]['generated'][:200]}...")
        print(f"参考答案: {results[i]['reference'][:200]}...")

if __name__ == "__main__":
    evaluate_model()
7. 训练监控和优化建议
监控指标：
训练损失：应该稳步下降

验证损失：应该低于或接近训练损失

Perplexity：数学模型的困惑度通常较高，但训练后会降低

优化建议：
逐步增加难度：先从简单数学题开始，再逐步增加复杂度

数据增强：对数学问题进行改写、增加多步推理

课程学习：按难度分级训练

集成多个数学数据集：GSM8K、MATH、AQUA等

硬件要求：
最低配置：1×RTX 3090 (24GB) 或 2×RTX 4090 (24GB×2)

推荐配置：4×A100 (80GB) 或 8×RTX 4090

使用QLoRA：可以在单张RTX 3090上训练

这个代码框架提供了完整的SFT训练流程，你可以根据具体的数学任务和数据集进行调整优化。

上面用的optimizer是哪一个
在上面的代码中，我使用了 paged_adamw_8bit 优化器。这是专门为4-bit/8-bit量化训练设计的高效优化器。

优化器详解
1. 使用的优化器
python
optim: str = field(
    default="paged_adamw_8bit",
    metadata={"help": "优化器"}
)
2. 为什么选择这个优化器？
对于量化训练（QLoRA）：

paged_adamw_8bit：专门为8-bit量化AdamW设计，通过分页机制管理优化器状态，显著减少显存占用

由bitsandbytes库提供，支持高效的4-bit/8-bit训练

在处理大模型时，相比标准AdamW可以减少约50%的显存占用

3. 其他可用的优化器选项
根据你的硬件和需求，可以选择不同的优化器：

python
# 完整的优化器选择列表
OPTIMIZER_CHOICES = [
    "adamw_hf",          # 标准AdamW (HuggingFace实现)
    "adamw_torch",       # PyTorch的AdamW
    "adamw_8bit",        # 8-bit AdamW (bitsandbytes)
    "paged_adamw_8bit",  # 分页8-bit AdamW (推荐用于QLoRA)
    "paged_adamw_32bit", # 分页32-bit AdamW
    "adamw_bnb_8bit",    # 8-bit AdamW的别名
    "sgd",               # 随机梯度下降
    "adagrad",            # AdaGrad
    "adafactor",          # 内存高效的AdaFactor
    "lion",              # 新的高效优化器
]
4. 不同场景下的优化器推荐
场景1：QLoRA训练（4-bit量化）
python
# 推荐：paged_adamw_8bit
training_args = TrainingArguments(
    optim="paged_adamw_8bit",  # 显存效率最高
    # ... 其他参数
)
场景2：全参数微调（足够显存）
python
# 推荐：adamw_torch 或 adamw_hf
training_args = TrainingArguments(
    optim="adamw_torch",  # 标准AdamW，更稳定
    # ... 其他参数
)
场景3：内存受限环境
python
# 推荐：adafactor（最节省内存）
training_args = TrainingArguments(
    optim="adafactor",  # 内存效率极高，适合超大模型
    # ... 其他参数
)
5. 优化器参数配置
可以在训练参数中添加优化器特定配置：

python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    optim="paged_adamw_8bit",
    
    # AdamW优化器参数
    learning_rate=2e-4,
    weight_decay=0.01,
    adam_beta1=0.9,      # β1参数
    adam_beta2=0.999,    # β2参数
    adam_epsilon=1e-8,   # 防止除零的小常数
    
    # 学习率调度
    lr_scheduler_type="cosine",  # 余弦退火
    warmup_ratio=0.03,           # 预热比例
    
    # 梯度相关
    max_grad_norm=1.0,    # 梯度裁剪
    gradient_accumulation_steps=4,
    
    # ... 其他参数
)
6. 自定义优化器
如果需要更细粒度的控制，可以自定义优化器：

python
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

# 自定义优化器
def get_optimizer(model, training_args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon
    )
    
    return optimizer

# 在Trainer中使用
class CustomTrainer(Trainer):
    def create_optimizer(self):
        return get_optimizer(self.model, self.args)

# 使用自定义Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    # ... 其他参数
)
7. 针对数学训练的优化器建议
对于数学推理任务，推荐以下配置：

python
# 数学SFT训练推荐配置
math_training_args = {
    "optim": "paged_adamw_8bit",      # 量化训练的高效选择
    "learning_rate": 1e-4,            # 数学任务需要较小的学习率
    "lr_scheduler_type": "cosine",    # 平滑的学习率衰减
    "warmup_ratio": 0.05,             # 足够的预热
    "weight_decay": 0.01,             # 适中的权重衰减防止过拟合
    "max_grad_norm": 1.0,             # 防止梯度爆炸
}
8. 监控优化器状态
python
# 在训练回调中监控优化器状态
from transformers import TrainerCallback

class OptimizerMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 监控学习率
            if "learning_rate" in logs:
                print(f"当前学习率: {logs['learning_rate']:.2e}")
            
            # 监控梯度范数
            if "grad_norm" in logs:
                print(f"梯度范数: {logs['grad_norm']:.4f}")
            
            # 监控优化器状态内存
            if state.optimizer is not None:
                optimizer_memory = sum(
                    p.numel() * p.element_size() 
                    for group in state.optimizer.param_groups 
                    for p in group['params']
                ) / 1024**3  # 转换为GB
                print(f"优化器状态内存: {optimizer_memory:.2f} GB")

# 添加到Trainer
trainer = Trainer(
    callbacks=[OptimizerMonitorCallback()],
    # ... 其他参数
)
总结
对于你的 Qwen2.5-Math-1.5B SFT训练：

使用 paged_adamw_8bit 是最佳选择，特别是当你使用4-bit QLoRA时

学习率建议在 1e-5 到 5e-4 之间调整

结合余弦退火学习率调度器

添加适度的权重衰减 (0.01) 来防止数学任务上的过拟合

如果你有足够的显存进行全参数微调，也可以考虑使用标准的 adamw_torch，它在某些情况下可能更稳定。

