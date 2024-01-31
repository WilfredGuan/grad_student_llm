from typing import Any
from tqdm import tqdm
import torch
import numpy as np
import time
import re
import os
import json
from model.model_hook import ModelOutputExporter


class ModelEvaluatorBase:
    def __init__(self, model, tokenizer, data, eval_args, logger, accelerator):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.eval_args = eval_args
        self.logger = logger
        self.accelerator = accelerator

        self.batch_size = self.eval_args.batch_size
        self.temperature = self.eval_args.temperature
        self.top_p = self.eval_args.top_p
        self.n_samples = self.eval_args.n_samples
        self.k = self.eval_args.k
        self.max_new_tokens = self.eval_args.max_new_tokens

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
        prompt_split = np.array_split(range(len(self.data)), dp_size)[dp_rank]
        indices = [
            i * self.n_samples + j for i in prompt_split for j in range(self.n_samples)
        ]
        indices_len = len(indices)
        processed_num = 0
        file_name = f"rank_{dp_rank}_eval.txt"
        start_time = time.time()
        local_exporter = ModelOutputExporter(dp_rank, self.model)

        # each device process a part of the data, per batch
        for idx in range(0, indices_len, self.batch_size):
            prompt_list = []
            prompt_lens = []
            ground_truth_list = []

            whole_generation_list = []
            generation_only_list = []
            cleaned_answer_list = []

            message_to_tokenizer = []

            local_exporter.outputs = [{"outputs": {}, "result": ""}]

            # for each example in the batch
            for example in self.data[indices[idx] : indices[idx] + self.batch_size]:
                prompt = example["prompt"].strip()
                prompt_list.append(prompt)
                # print("example answer:", example["answer"].strip())
                ground_truth_list.extend(
                    self._cleanning_output([example["answer"].strip()])
                )
                # print("prompt", prompt)
                tmp = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    max_length=1024,
                    # padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_lens.append(len(tmp))
                message_to_tokenizer.extend(np.array(tmp))
            # print("Rank", dp_rank, "message_to_tokenizer", message_to_tokenizer)
            # print("shape of message_to_tokenizer", message_to_tokenizer[0].shape)
            input_ids = torch.tensor(np.array(message_to_tokenizer)).to(
                self.accelerator.device
            )

            raw_output_list, gen_only_output_list = self._eval_gen(
                input_ids, prompt_lens
            )
            whole_generation_list.extend(raw_output_list)
            generation_only_list.extend(gen_only_output_list)

            cleaned_answer_list.extend(self._cleanning_output(generation_only_list))
            # log in file
            for i in range(self.batch_size):
                self.logger.log_in_jsonl(
                    name="GSM8K_" + str(dp_rank),
                    message={
                        "Prompt": f"{prompt_list[i]}",
                        "Cleaned Generation": f"{cleaned_answer_list[i]}",
                        "Ground Truth": f"{ground_truth_list[i]}",
                        "Generation Only": f"{generation_only_list[i]}",
                        "Raw Generation": f"{whole_generation_list[i]}",
                    },
                )
                local_exporter.export_to_jsonl(
                    self.logger.get_log_dir()
                    + "/Hook_GSM8K_"
                    + str(dp_rank)
                    + "_log.jsonl"
                )
            processed_num += len(prompt_list)
            avg_time = (time.time() - start_time) / processed_num * self.batch_size
            if processed_num == indices_len:
                print("Evaluation finished on rank:", dp_rank)
            else:
                print(
                    f"Rank {dp_rank}:{processed_num} / {indices_len} "
                    f"avg_time per batch: {avg_time:.2f} s "
                    f"still need {((indices_len-processed_num)//self.batch_size+1)*avg_time/60:.2f} minutes"
                )
        local_exporter.remove_hooks()
        self.accelerator.wait_for_everyone()
        self._check_correctness()
        self.accelerator.wait_for_everyone()
        return

    @torch.no_grad()
    def _eval_gen(self, tokenized_input_list, prompt_lens):
        # message = [{"role": "user", "content": message}]
        # model_inputs = self.tokenizer.apply_chat_template(
        #     message,
        #     max_length=1024,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt",
        # ).to(self.accelerator.device)
        if self.temperature != 0:
            generated_ids = self.model.generate(
                input_ids=tokenized_input_list,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generated_ids = self.model.generate(
                input_ids=tokenized_input_list,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw_output_list = []
        gen_only_output_list = []
        for local_idx, text in enumerate(generated_ids):
            raw_output = generated_ids[local_idx]
            raw_output = self.tokenizer.decode(
                raw_output,
                skip_special_tokens=True,
            )
            raw_output_list.append(raw_output)

            match = re.search(r"\[/INST\]", raw_output)
            end = match.end()
            gen_only_output = raw_output[end:]
            gen_only_output_list.append(gen_only_output)
        return raw_output_list, gen_only_output_list

    def _cleanning_output(self, output):
        cleaned_output = []
        for i in output:
            pattern = r"#### (\d+)"
            match = re.search(pattern, i)
            if match:
                cleaned_output.append(match.group(1))
            else:
                cleaned_output.append("No match")
        return cleaned_output

    def _check_correctness(self):
        if self.accelerator.is_local_main_process:
            print("Checking correctness...")
            log_dir = self.logger.get_log_dir()
            log_file_path = log_dir + "/GSM8K_final_log.jsonl"
            hook_file_path = log_dir + "/Hook_GSM8K_final_log.jsonl"
            log_file = open(log_file_path.replace(" ", ""), "w")
            hook_file = open(hook_file_path.replace(" ", ""), "w")

            score = 0
            num_examples = 0
            correct_examples = 0
            for i in range(self.accelerator.num_processes):
                log_file_i = open(log_dir + f"/GSM8K_{i}_log.jsonl", "r")
                hook_file_i = open(log_dir + f"/Hook_GSM8K_{i}_log.jsonl", "r")
                for line1, line2 in zip(log_file_i, hook_file_i):
                    num_examples += 1
                    line1 = json.loads(line1)
                    line2 = json.loads(line2)
                    if line1["Cleaned Generation"] == line1["Ground Truth"]:
                        line1["Correctness"] = "True"
                        line2["result"] = "True"
                        correct_examples += 1
                    else:
                        line1["Correctness"] = "False"
                        line2["result"] = "False"
                    log_file.write(json.dumps(line1) + "\n")
                    hook_file.write(json.dumps(line2) + "\n")
                log_file_i.close()
                hook_file_i.close()
                os.remove(log_dir + f"/GSM8K_{i}_log.jsonl")
                os.remove(log_dir + f"/Hook_GSM8K_{i}_log.jsonl")
            log_file.close()
            hook_file.close()
            score = correct_examples / num_examples
            print("Final score:", score)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("start evaluation on GMS8K dataset")
        self._eval()


if __name__ == "__main__":
    eval = GSM8K(None, None, None, None, None, None)
    eval._check_correctness()
