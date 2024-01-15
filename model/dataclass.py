from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-6.7b-instruct"
    )
    torch_dtype: str = field(default="float16")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainArguments(transformers.TrainingArguments):
    train_flag: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class EvalArguments:
    pass
