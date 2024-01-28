from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from transformers import Trainer
import re


class ModelTrainerBase:
    def __init__(
        self,
        model,
        device,
        logger,
        train_eval_args,
    ):
        self.train_eval_args = train_eval_args

        self.model = model
        self.device = device
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

    def huamn_eval(self, test_set):
        accuracy = 0
        queries = []
        raw_answer = []
        numeric_answer = []

        for sample in test_set:
            queries.append(sample["question"])
            raw_answer.append(sample["answer"])
            numeric_pattern = re.compile(r"####\s*(\d+)")
            match = numeric_pattern.search(sample["answer"])
            if match:
                number = int(match.group(1))
                numeric_answer.append(number)
            else:
                numeric_answer.append(None)

        # Log the basic status of datasets
        # Note that if queries, raw_anwer and numeric_answer are not in the same length, something is wrong.
        dataset_log = f"Split queries - \n[Test set] {len(test_set)}\n[Query number] {len(queries)}\n[Raw answer number] {len(raw_answer)}\n[Numeric answer number] {len(numeric_answer)}\n"
        self.logger.log_message_txt(dataset_log)

        # construct prompt based on the queries
        prompted_data = self.construct_prompt(queries)
        numeric_pattern = re.compile(r"####\s*(\d+)")

        for data in range(len(prompted_data)):
            print(f"Running on {data+1} / {len(queries)}")
            print("Prompted data:", prompted_data[data])
            result = self.model.model_gen(prompted_data[data])
            match = numeric_pattern.search(result)
            print(f"Here is matched answer: {match}")
            print(f"Here is true answer: {numeric_answer[data]}")
            acc_temp = False
            if match:
                number = int(match.group(1))
                if number == numeric_answer[data]:
                    accuracy += 1
                    acc_temp = True

            result_log = (
                "-" * 10
                + f"\nExample index: {data}\n"
                + f"[Raw question] {queries[data]}\n[Prompted question] {prompted_data[data]}\n[True numeric answer] {numeric_answer[data]}\n[Model answer] {result}\n[True or false] {acc_temp}\n"
            )
            self.logger.log_message_txt(result_log)

        accuracy = accuracy / len(queries)
        print(f"Accuracy: {accuracy}")
        self.logger.log_message_txt(f"Accuracy: {accuracy}")
        return accuracy

    def construct_prompt(self, question):
        prompt_var_1 = f"You are a high school math teacher, try to solve one problem step by step according to examples. Annotate your final numeric answer as '#### YOUR_NUMBER'.\n"
        prompt_set = []

        # shot_num - how many examples we want in the prompted queries
        shot_num = 0
        # choose a basic instruction to the question
        using_prompt = prompt_var_1

        for i in question:
            prompt = {"role": "user", "content": ""}
            # considering that the model is relatively small and it is hard to digest long text
            prompt["content"] = using_prompt + "\n" + i + "\n"
            prompt_set.append([prompt])

        construction_length = len(prompt_set)
        # print(prompt_set[0])
        construction_log = f"Construct - [length] {construction_length} \n[Number of shots] {shot_num}\n[Prompt used] {using_prompt}"
        self.logger.log_message_txt(construction_log)
        print(construction_log)
        print("Construction complete.")

        return prompt_set

    def compute_loss(self):
        pass
