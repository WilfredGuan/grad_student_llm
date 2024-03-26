import json
import re
import random


class Constructor:
    def __init__(self, data, *kwargs):
        self.data = data

    def get_data(self):
        pass

    def _get_synthetic_data(self):
        pass


class GSM8KConstructor(Constructor):
    def __init__(self, data, data_args, *kwargs):
        super().__init__(data=data)

        self.data_args = data_args
        self.data = self._construct(data)

    def _construct(self, data):
        print("Constructing GSM8K data...")

        # template = "You are a math teacher thinking step by step. Your question is: {} \n\n Your final answer must be like this #### NUMBER"
        template = "You should answer the following question step by step. Replace your final numeric answer in this format #### YOUR_NUMBER.\nQuestion: {}\n\n Your answer is:"

        # zero-shot
        if self.data_args.construction_mode == "zero-shot":
            prompt_list = []
            for sample in data:
                prompt = template.format(sample["question"])
                prompt = {
                    "prompt": prompt,
                    "question": sample["question"],
                    "answer": sample["answer"],
                }
                prompt_list.append(prompt)

        # n-shot
        if self.data_args.construction_mode == "n-shot":
            n_shot = self.data_args.n_shot
            train_data = data["train"]
            test_data = data["test"]
            # containing the constructed data
            prompt_list = []

            # for each data
            for sample in test_data:
                # separately storage the n-shot prompts & testing prompt
                res_sample = {"n_shot": [], "prompt": {}}

                # start constructing n-shot prompts
                for i in range(n_shot):
                    n_shot_sample = {
                        "prompt": "",
                        "question": "",
                        "answer": "",
                    }
                    # randomly select n examples from train_data
                    random_shot = random.choice(train_data)
                    n_shot_sample["prompt"] = template.format(random_shot["question"])
                    n_shot_sample["question"] = random_shot["question"]
                    n_shot_sample["answer"] = random_shot["answer"]
                    res_sample["n_shot"].append(n_shot_sample)

                # add test prompt and answer
                prompt_sample = {
                    "prompt": "",
                    "question": "",
                    "answer": "",
                }
                prompt_sample["prompt"] = template.format(sample["question"])
                prompt_sample["question"] = sample["question"]
                prompt_sample["answer"] = sample["answer"]
                res_sample["prompt"] = prompt_sample
                prompt_list.append(res_sample)

        return prompt_list

    # def _n_shot_construct(self, data):
    #     """
    #     input structure: {"train": train_data, "test": test_data}
    #     """
    #     train_data = self._construct(data["train"])
    #     test_data = self._construct(data["test"])

    #     result = []
    #     for sample in test_data:
    #         res_sample = {
    #             "prompt": [],
    #             "original_question_train": [],
    #             "original_question_test": [],
    #             "answer_in_steps_train": [],
    #             "answer_in_steps_test": [],
    #         }
    #         for i in range(self.n):
    #             # randomly select n examples from train_data
    #             random_shot = random.choice(train_data)
    #             # add n-shot prompt
    #             res_sample["prompt"].extend(random_shot["prompt"])
    #             # add original question
    #             res_sample["original_question_train"].extend(
    #                 random_shot["original_question"]
    #             )
    #             # add n-shot answer_in_steps
    #             res_sample["answer_in_steps_train"].extend(random_shot["answer"])
    #         assert (
    #             len(res_sample["prompt"]) == self.n
    #             and len(res_sample["answer_in_steps_train"]) == self.n
    #         ), "n-shot construction failed."
    #         # add test prompt and answer
    #         res_sample["prompt"].extend(sample["prompt"])
    #         # add test original question and answer_in_steps
    #         res_sample["original_question_test"] = sample["original_question"]
    #         res_sample["answer_in_steps_test"] = sample["answer"]
    #         # print(res_sample)
    #         result.append(res_sample)
    #     return result

    # def _cot_construct(self, data):
    #     pass


class Step2KnowledgeConstructor(Constructor):
    def __init__(self, data, data_args, *kwargs):
        super().__init__(data=data)
        self.data_args = data_args
        self.data = self._construct(data)

    def _construct(self, data):
        print("Constructing Step 2 data...")
        # template = "You are a math teacher thinking step by step. Your question is: {} \n\n Your final answer must be like this #### NUMBER"
        ques_template = "You need to understand the following question and an example answer. According to your knowledge, rethink the and conclude whether the target answer is [Correct] or [Wrong].\nQuestion: {}\nExample Answer: {}\nTarget Answer: {}"
        prompt_list = []
        for sample in data:
            sample["ques"] = sample["ques"][0]
            sample["ans"] = re.sub(r"####.*$", "", sample["ans"][0])
            sample["generated_ans"] = re.sub(r"####.*$", "", sample["generated_ans"])
            prompt = ques_template.format(
                sample["ques"], sample["ans"], sample["generated_ans"]
            )
            prompt_list.append(
                {
                    "prompt": [prompt],
                    "ques": sample["ques"],
                    "example_ans": sample["ans"],
                    "target_ans": sample["generated_ans"],
                    "ground_truth_correctness": sample["ground_truth_correctness"],
                }
            )
        return prompt_list
