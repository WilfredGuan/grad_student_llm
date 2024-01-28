import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.dataclass import *
from data.dataloader import *
from data.constructor import *
from model.trainer import *
from model.evaluator import *


class ModelBase:
    def __init__(
        self,
        model_arguemnts,
        data_arguments,
        logger,
        accelerator,
        train_arguments=None,
        eval_arguments=None,
    ):
        self.model_args = model_arguemnts
        self.data_args = data_arguments
        self.train_args = train_arguments
        self.eval_args = eval_arguments
        self.logger = logger

        # print("Cuda availability:", torch.cuda.is_available())
        self.model, self.tokenizer = self._load_from_hf(accelerator)

        self.dataloader = eval(self.data_args.dataloader)(
            self.data_args.data_path, batch_size=1
        )

        if self.train_args != None:
            self.trainer = eval(self.train_args.trainer_name)(
                model=self,
                device=self.device,
                logger=self.logger,
                train_args=self.train_args,
                accelerator=accelerator,
            )
        if self.eval_args != None:
            self.evaluator = eval(self.eval_args.evaluator_name)(
                model=self.model,
                dataloader=self.dataloader,
                eval_args=self.eval_args,
                logger=self.logger,
                accelerator=accelerator,
            )

    def model_gen(self, messages):
        model_inputs = self.preprocess(messages)
        generated_ids = self.model.generate(
            input_ids=model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
        )
        decoded = self.tokenizer.decode(
            generated_ids["sequences"][0][len(model_inputs[0]) :],
            skip_special_tokens=True,
        )
        return decoded

    def preprocess(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
        )
        return input_ids

    def train(self):
        assert (
            self.train_args != None
        ), "Training arguments are not provided. Please provide training arguments to activate this function."
        self.trainer()
        self.save(self.train_args.save_path)

    def eval(self):
        assert (
            self.eval_args != None
        ), "Evaluation arguments are not provided. Please provide evaluation arguments to activate this function."
        self.evaluator()

    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def _load_from_hf(self, accelerator):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=self.model_args.torch_dtype,
            device_map=accelerator.device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=self.model_args.torch_dtype,
            device_map=accelerator.device,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer
