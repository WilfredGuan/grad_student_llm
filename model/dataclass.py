from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
import torch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    torch_dtype: any = field(default=torch.bfloat16)
    device: str = field(default="cpu")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dataloader: str = field(default="GSM8KLoader")
    constructor: str = field(default="GSM8KConstructor")


@dataclass
class TrainArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(
        default="./output_dir",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    cache_dir: Optional[str] = field(default="./cache_dir")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class EvalArguments:
    evaluator_name: str = field(default="ModelEvaluatorBase")
