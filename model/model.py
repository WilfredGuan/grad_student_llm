import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM

from model.trainer import *
from model.dataclass import *


class ModelBase:
    def __init__(self, ModelArguments, DataArguments, TrainingArguments):
        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )

        (
            self.model_args,
            self.data_args,
            self.training_args,
        ) = parser.parse_args_into_dataclasses()
        # print("Cuda availability:", torch.cuda.is_available())

        self.model, self.tokenizer = self.load_from_hf()

        # self.data

        if self.training_args.train_flag == True:
            self.model_train = ModelTrainer(
                self.model,
                self.tokenizer,
                self.device,
                self.datacollator,
                self.logger,
                self.ModelArguments,
                self.DataArguments,
                self.TrainingArguments,
            )

    def model_gen(self, messages):
        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", padding=True
        )

        model_inputs = encodeds.to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(
            model_inputs, max_new_tokens=1000, do_sample=True
        )
        decoded = self.tokenizer.batch_decode(generated_ids)
        # print(decoded[0])
        return decoded[0]

    def preprocess(self):
        pass

    def train(self):
        if self.training_args.train_flag == False:
            raise Exception("Training flag is set to False.")
        else:
            self.model_train.train()

    def test(self, test_loader, epoch):
        pass

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_from_hf(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            model_max_length=self.training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        return model, tokenizer

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_criterion(self):
        return self.criterion

    def get_device(self):
        return self.device
