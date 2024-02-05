from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
import torch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.2"
    )
    torch_dtype: any = field(default=torch.float16)
    device: str = field(default="cpu")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dataloader: str = field(default="GSM8KLoader")
    split_ratio: float = field(default=None)
    split_validation: bool = field(default=False)
    train_path: str = field(default=None)
    test_path: str = field(default=None)
    val_path: str = field(default=None)
    if split_ratio is None:
        assert train_path is not None, "Train path or name is required."
        assert test_path is not None, "Test path or name is required."
        assert val_path is not None, "Validation path or name is required."

    constructor: str = field(default="GSM8KConstructor")
    construction_mode: str = field(default="zero-shot")
    n_shot: int = field(default=0)
    if construction_mode == "n-shot":
        assert n_shot > 0, "You should set n_shot = n for n-shot construction."

    def __post_init__(self):
        valid_modes = ["0-shot", "n-shot", "cot"]
        if self.construction_mode not in valid_modes:
            raise ValueError(
                f"construction_mode must be one of {valid_modes}, got '{self.construction_mode}'"
            )


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
    n_samples: int = field(default=1)
    k: int = field(default=1)
    temperature: float = field(default=0)
    top_p: float = field(default=0.95)
    max_new_tokens: int = field(default=200)
    batch_size: int = field(default=1)
