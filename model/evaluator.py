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


class GSM8KEval(ModelEvaluatorBase):
    def __init__(
        self, model, tokenizer, data, data_args, eval_args, logger, accelerator
    ):
        super().__init__(model, tokenizer, data, eval_args, logger, accelerator)

        # check data mode
        self.data = data
        self.data_args = data_args
        if self.data_args.construction_mode == "zero-shot":
            self.mode = "zero-shot"
        if self.data_args.construction_mode == "n-shot":
            self.n = self.data_args.n_shot
            self.mode = str(self.n) + "-shot"

    def _eval(self):
        # split the data into different parts, according to the number of devices
        dp_rank = self.accelerator.process_index
        dp_size = self.accelerator.num_processes
        prompt_split = np.array_split(range(len(self.data)), dp_size)[dp_rank]

        # applying n_samples
        indices = [
            i * self.n_samples + j for i in prompt_split for j in range(self.n_samples)
        ]

        if self.n_samples > 1:
            prompt_split = [x for x in prompt_split for _ in range(self.n_samples)]

        # ------
        # some process control / visualization information
        # data size for each device
        indices_len = len(indices)
        processed_num = 0
        start_time = time.time()
        # ------
        local_exporter = ModelOutputExporter(dp_rank, self.model)

        # each device process a part of the data, per batch
        for idx in range(0, indices_len, self.batch_size):
            """
            Two kind of construction mode: zero-shot or n-shot
            zero-shot:
            input structure: [{
                    "prompt": prompt,
                    "question": sample["question"],
                    "answer": sample["answer"],
                }, ...]

            n-shot:
            input structure: [
                {
                "n_shot": [{
                    "prompt": prompt,
                    "question": sample["question"],
                    "answer": sample["answer"],
                }, ...],
                "prompt": {
                    "prompt": prompt,
                    "question": sample["question"],
                    "answer": sample["answer"],
                },
                }, ...]
            """
            prompt_list = []
            prompt_lens = []
            ground_truth_list = []
            whole_generation_list = []
            generation_only_list = []
            cleaned_answer_list = []
            message_to_tokenizer = []

            # for each example in the batch
            for example in self.data[indices[idx] : indices[idx] + self.batch_size]:

                # according to tokenizer.apply_chat_template
                chat_template = []

                # for zero-shot
                if self.mode == "zero-shot":
                    chat_sample = {"role": "user", "content": ""}
                    prompt = example["prompt"].strip()
                    prompt_list.append(prompt)
                    chat_sample["content"] = prompt
                    # print("example answer:", example["answer"].strip())
                    ground_truth_list.extend(
                        self._cleanning_output([example["answer"].strip()])
                    )
                # for n-shot/cot
                else:
                    for i in range(self.n):
                        chat_sample_user = {"role": "user", "content": ""}
                        chat_sample_assistant = {"role": "assistant", "content": ""}
                        # print("example:", example)

                        prompt = example["n_shot"][i]["prompt"].strip()
                        answer = example["n_shot"][i]["answer"].strip()

                        chat_sample_user["content"] = prompt
                        chat_sample_assistant["content"] = answer

                        chat_template.append(chat_sample_user)
                        chat_template.append(chat_sample_assistant)

                    chat_template.append(
                        {"role": "user", "content": example["prompt"]["prompt"].strip()}
                    )
                    # print("chat_template:", chat_template)
                    prompt_list.append(chat_template)
                    ground_truth_list.extend(
                        self._cleanning_output([example["prompt"]["answer"].strip()])
                    )

                # before this, we have converted the templates into chat format
                # all process control variables are already set
                tmp = self.tokenizer.apply_chat_template(
                    chat_template,
                    max_length=2048,
                    # padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_lens.append(len(tmp[0]))
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
            # processed dataset + answer
            for i in range(self.batch_size):
                message = {
                    # for n-shot problems, we have stored all the shots and the prompt in the prompt_list
                    "prompt": prompt_list[i],
                    "cleaned_answer": cleaned_answer_list[i],
                    "ground_truth": ground_truth_list[i],
                    "generation": generation_only_list[i],
                    "raw_generation": whole_generation_list[i],
                }
                if self.mode == "zero-shot":
                    message["question"] = example["question"]
                    message["answer"] = example["answer"]
                else:
                    message["n_shot"] = example["n_shot"]
                    message["prompt"] = example["prompt"]

                self.logger.log_in_jsonl(
                    name="GSM8K_"
                    + self.mode
                    + "_"
                    + self.data_args.task_mode
                    + "_"
                    + str(dp_rank),
                    message=message,
                )

                # hook
                local_exporter.export_to_jsonl(
                    self.logger.get_log_dir()
                    + "/Hook_GSM8K_"
                    + self.mode
                    + "_"
                    + self.data_args.task_mode
                    + "_"
                    + str(dp_rank)
                    + "_log.jsonl"
                )
            processed_num += len(prompt_list)
            avg_time = (time.time() - start_time) / processed_num * self.batch_size
            if processed_num == indices_len:
                print("Generation finished on rank:", dp_rank)
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

            match = list(re.finditer(r"\[/INST\]", raw_output))
            if match:
                last_match = match[-1]
                end = last_match.end()
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
            log_file_path = (
                log_dir
                + f"/GSM8K_{self.mode}_{self.data_args.task_mode}_final_log.jsonl"
            )
            hook_file_path = (
                log_dir
                + f"/Hook_GSM8K_{self.mode}_{self.data_args.task_mode}_final_log.jsonl"
            )
            log_file = open(log_file_path.replace(" ", ""), "w")
            hook_file = open(hook_file_path.replace(" ", ""), "w")

            score = 0
            num_examples = 0
            correct_examples = 0
            for i in range(self.accelerator.num_processes):
                # prepare the final log
                log_file_i = open(
                    log_dir
                    + f"/GSM8K_{self.mode}_{self.data_args.task_mode}_{i}_log.jsonl",
                    "r",
                )
                hook_file_i = open(
                    log_dir
                    + f"/Hook_GSM8K_{self.mode}_{self.data_args.task_mode}_{i}_log.jsonl",
                    "r",
                )
                # start comparison
                if self.n_samples > 1:
                    n_sample_correct = 0
                    line1_list = []
                    line2_list = []

                    for line1, line2 in zip(log_file_i, hook_file_i):
                        num_examples += 1
                        line1 = json.loads(line1)
                        line2 = json.loads(line2)

                        if line1["cleaned_answer"] == line1["ground_truth"]:
                            n_sample_correct += 1
                            line1["correctness"] = "True"
                            line2["correctness"] = "True"
                            correct_examples += 1
                        else:
                            line1["correctness"] = "False"
                            line2["correctness"] = "False"

                        line1_list.append(line1)
                        line2_list.append(line2)

                        if len(line1_list) == 10:
                            for line1, line2 in zip(line1_list, line2_list):
                                line1["avg_correctness"] = n_sample_correct / 10
                                line2["avg_correctness"] = n_sample_correct / 10
                                log_file.write(json.dumps(line1) + "\n")
                                hook_file.write(json.dumps(line2) + "\n")
                            line1_list = []
                            line2_list = []
                            n_sample_correct = 0
                else:
                    for line1, line2 in zip(log_file_i, hook_file_i):
                        num_examples += 1
                        line1 = json.loads(line1)
                        line2 = json.loads(line2)
                        if line1["cleaned_answer"] == line1["ground_truth"]:
                            line1["correctness"] = "True"
                            line2["result"] = "True"
                            correct_examples += 1
                        else:
                            line1["Correctness"] = "False"
                            line2["result"] = "False"
                        log_file.write(json.dumps(line1) + "\n")
                        hook_file.write(json.dumps(line2) + "\n")
                log_file.flush()
                hook_file.flush()
                log_file_i.close()
                hook_file_i.close()
                os.remove(log_dir + f"/GSM8K_{self.mode}_{i}_log.jsonl")
                os.remove(log_dir + f"/Hook_GSM8K_{self.mode}_{i}_log.jsonl")
            log_file.close()
            hook_file.close()
            score = correct_examples / num_examples
            print("Final score:", score)
            self.log_file.write(
                f"Task: {self.data_args.task_mode} on GSM8K dataset, {self.mode}\n"
                + "Final score: "
                + str(score)
                + " on "
                + str(num_examples)
                + " examples\n"
            )
            self.log_file.flush()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("start evaluation on GMS8K dataset")
        self._eval()


class Step2KnowledgeEvaluator(ModelEvaluatorBase):
    def __init__(self, model, tokenizer, data, eval_args, logger, accelerator):
        super().__init__(model, tokenizer, data, eval_args, logger, accelerator)

        self.mode = "step2knowledge-zero-shot"

    def _eval(self):
        # split the data into different parts, according to the number of devices
        dp_rank = self.accelerator.process_index
        dp_size = self.accelerator.num_processes
        prompt_split = np.array_split(range(len(self.data)), dp_size)[dp_rank]
        indices = [
            i * self.n_samples + j for i in prompt_split for j in range(self.n_samples)
        ]

        # ------
        # some process control / visualization information
        # data size for each device
        indices_len = len(indices)
        processed_num = 0
        start_time = time.time()
        # ------
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

            # for each example in the batch
            for example in self.data[indices[idx] : indices[idx] + self.batch_size]:

                chat_sample = {"role": "user", "content": ""}
                prompt = example["prompt"][0].strip()
                prompt_list.append(prompt)
                chat_sample["content"] = prompt
                ground_truth_list.append(example["ground_truth_correctness"].strip())
                chat_template = [chat_sample]
                tmp = self.tokenizer.apply_chat_template(
                    chat_template,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_lens.append(len(tmp[0]))
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
                # processed dataset + answer
                self.logger.log_in_jsonl(
                    name="GSM8K_" + self.mode + "_" + str(dp_rank),
                    message={
                        "Prompt": prompt_list[i],
                        "Cleaned Generation": cleaned_answer_list[i],
                        "Ground Truth": ground_truth_list[i],
                        "Generation Only": generation_only_list[i],
                        "Raw Generation": whole_generation_list[i],
                        "ques": example["ques"],
                        "example_ans": example["example_ans"],
                        "target_ans": example["target_ans"],
                    },
                )
                # hook
                local_exporter.export_to_jsonl(
                    self.logger.get_log_dir()
                    + "/Hook_GSM8K_"
                    + self.mode
                    + "_"
                    + str(dp_rank)
                    + "_log.jsonl"
                )
            processed_num += len(prompt_list)
            avg_time = (time.time() - start_time) / processed_num * self.batch_size
            if processed_num == indices_len:
                print("Generation finished on rank:", dp_rank)
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

            match = list(re.finditer(r"\[/INST\]", raw_output))
            if match:
                last_match = match[-1]
                end = last_match.end()
            gen_only_output = raw_output[end:]
            gen_only_output_list.append(gen_only_output)
        return raw_output_list, gen_only_output_list

    def _cleanning_output(self, output):
        cleaned_output = []
        for i in output:
            pattern = r"\[(.*?)\]"
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
            log_file_path = log_dir + f"/GSM8K_{self.mode}_final_log.jsonl"
            hook_file_path = log_dir + f"/Hook_GSM8K_{self.mode}_final_log.jsonl"
            log_file = open(log_file_path.replace(" ", ""), "w")
            hook_file = open(hook_file_path.replace(" ", ""), "w")

            score = 0
            num_examples = 0
            correct_examples = 0
            for i in range(self.accelerator.num_processes):
                log_file_i = open(log_dir + f"/GSM8K_{self.mode}_{i}_log.jsonl", "r")
                hook_file_i = open(
                    log_dir + f"/Hook_GSM8K_{self.mode}_{i}_log.jsonl", "r"
                )
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
                log_file.flush()
                hook_file.flush()
                log_file_i.close()
                hook_file_i.close()
                os.remove(log_dir + f"/GSM8K_{self.mode}_{i}_log.jsonl")
                os.remove(log_dir + f"/Hook_GSM8K_{self.mode}_{i}_log.jsonl")
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
