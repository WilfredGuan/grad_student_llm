from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer


class ModelTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        datacollator,
        logger,
        ModelArguments,
        DataArguments,
        TrainArguments,
    ):
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainArguments)
        )

        (
            self.model_args,
            self.data_args,
            self.training_args,
        ) = parser.parse_args_into_dataclasses()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.datacollator = datacollator
        self.logger = logger

        self.IGNORE_INDEX = -100
        self.EOT_TOKEN = "<|EOT|>"
        self.logger.log_message_txt(
            "ModelTrainer initialized\nUsing {self.EOT_TOKEN} as EOT token\nUsing {self.IGNORE_INDEX} as IGNORE_INDEX\n".format(
                self=self
            )
        )

    def train(self):
        if self.training_args.local_rank == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.training_args.n_gpu = torch.cuda.device_count()

        token_message = """PAD Token: {self.tokenizer.pad_token}, {self.tokenizer.pad_token_id}\n
            BOS Token: {self.tokenizer.bos_token}, {self.tokenizer.bos_token_id}\n
            EOS Token: {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id}\n""".format(
            self=self
        )
        self.logger.log_message_txt(token_message)

        if self.training_args.local_rank == 0:
            print(
                "Train model from {} initialized".format(
                    self.model_args.model_name_or_path
                )
            )

    def compute_loss(self):
        pass
