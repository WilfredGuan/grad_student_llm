from typing import Any
from tqdm import tqdm
import torch
import numpy as np
import time


class ModelEvaluatorBase:
    def __init__(self, model, tokenizer, data, eval_args, logger, accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.evaluator_args = eval_args
        self.logger = logger
        self.accelerator = accelerator
        if self.evaluator_args.n_samples:
            self.n_samples = self.evaluator_args.n_samples
        if self.evaluator_args.k:
            self.k = self.evaluator_args.k
        if self.__class__.__name__ == "ModelEvaluatorBase":
            print("You are using ModelEvaluatorBase, which is a dummy evaluator.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _eval(self):
        pass

    def _cleanning_output(self, output):
        pass

    def _check_correctness(self, output, sample):
        pass

    def _calculate_final_score(self):
        pass


class HumanEval(ModelEvaluatorBase):
    def __init__(self, model, tokenizer, data, eval_args, logger, accelerator):
        super().__init__(model, tokenizer, data, eval_args, logger, accelerator)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("On Call")

    def _eval(self):
        pass

    def _cleanning_output(self, output):
        pass

    def _check_correctness(self, output, sample):
        pass

    def _calculate_final_score(self):
        pass


class GSM8K(ModelEvaluatorBase):
    def __init__(self, model, tokenizer, data, eval_args, logger, accelerator):
        super().__init__(model, tokenizer, data, eval_args, logger, accelerator)

    def _eval(self):
        dp_rank = self.accelerator.process_index
        dp_size = self.accelerator.num_processes
        prompt_split = np.array_split(self.data, dp_size)[dp_rank]
        indices = [
            i * self.n_samples + j for i in prompt_split for j in range(self.n_samples)
        ]
        indices_len = len(indices)
        processed_num = 0
        file_name = f"rank_{dp_rank}_eval.txt"
        start_time = time.time()
        for idx in range(0, indices_len):
            prompt_list = []
            prompt_lens = []
            original_prompt_list = []
            original_prompt_lens = []
            taskid = []
            for example in tqdm(self.data[indices[idx] : indices[idx] + 1]):
                prompt = example["prompt"]
                prompt_list.append(prompt)
                prompt_lens.append(len(prompt))
                original_prompt_list.append(example["original_prompt"])
                original_prompt_lens.append(len(example["original_prompt"]))
                taskid.append(example["taskid"])
            output = self._eval_gen(example["prompt"])
            self._cleanning_output(output)
            self._check_correctness(output, example)

    @torch.no_grad()
    def _eval_gen(self, message):
        message = [{"role": "user", "content": message}]
        model_inputs = self.tokenizer.apply_chat_template(
            message,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.accelerator.device)
        generated_ids = self.model.generate(
            input_ids=model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        decoded = self.tokenizer.decode(
            generated_ids["sequences"][0][len(model_inputs[0]) :],
            skip_special_tokens=True,
        )
        return decoded

    def _cleanning_output(self, output):
        pass

    def _check_correctness(self, output, sample):
        pass

    def _calculate_final_score(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("start evaluation on GMS8K dataset")
        self._eval()
