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
    """
    data_args -> direct apply to GSM8KLoader
    write some inner logic for mode detection
    expecially for GSM8K -> zero-shot, n-shot, cot
    zero-shot -> just return data

    only-difference is the data_args.construction_mode:
    n-shot -> return train & test data, and revise the Constructor correspondingly
    cot -> return train, test;
    """

    def __init__(self, data, data_args, *kwargs):
        super().__init__(data=data)

        self.data_args = data_args
        if self.data_args.construction_mode == "zero-shot":
            self.data = self._construct(data)
        if (
            self.data_args.construction_mode == "n-shot"
            or self.data_args.construction_mode == "cot"
        ):
            self.n = self.data_args.n_shot
            self.data = self._n_shot_construct(data)

    def _construct(self, data):
        print("Constructing GSM8K data...")
        # template = "You are a math teacher thinking step by step. Your question is: {} \n\n Your final answer must be like this #### NUMBER"
        template = "You are a math teacher thinking step by step. Replace your numeric answers to YOUR_NUMBER in this format #### YOUR_NUMBER.\n\nYour question is: {}\n\n Your answer is:"
        prompt_list = []
        for sample in data:
            prompt = template.format(sample["question"])
            prompt = {
                "prompt": [prompt],
                "original_question": [sample["question"]],
                "answer": [sample["answer"]],
            }
            prompt_list.append(prompt)
        # print("GSM8K data constructed.")
        # print("Number of examples:", len(prompt_list))
        # print("Example:", prompt_list[0])
        return prompt_list

    def _n_shot_construct(self, data):
        """
        input structure: {"train": train_data, "test": test_data}
        """
        train_data = self._construct(data["train"])
        test_data = self._construct(data["test"])

        result = []
        for sample in test_data:
            res_sample = {
                "prompt": [],
                "original_question_train": [],
                "original_question_test": [],
                "answer_in_steps_train": [],
                "answer_in_steps_test": [],
            }
            for i in range(self.n):
                # randomly select n examples from train_data
                random_shot = random.choice(train_data)
                # add n-shot prompt
                res_sample["prompt"].extend(random_shot["prompt"])
                # add original question
                res_sample["original_question_train"].extend(
                    random_shot["original_question"]
                )
                # add n-shot answer_in_steps
                res_sample["answer_in_steps_train"].extend(random_shot["answer"])
            assert (
                len(res_sample["prompt"]) == self.n
                and len(res_sample["answer_in_steps_train"]) == self.n
            ), "n-shot construction failed."
            # add test prompt and answer
            res_sample["prompt"].extend(sample["prompt"])
            # add test original question and answer_in_steps
            res_sample["original_question_test"] = sample["original_question"]
            res_sample["answer_in_steps_test"] = sample["answer"]
            # print(res_sample)
            result.append(res_sample)
        return result

    def _cot_construct(self, data):
        pass


class Step2Constructor(Constructor):
    def __init__(self, data, data_args, *kwargs):
        super().__init__(data=data)
        self.data_args = data_args
        self.data = self._construct(data)

    def _construct(self, data):
        print("Constructing Step 2 data...")
        # template = "You are a math teacher thinking step by step. Your question is: {} \n\n Your final answer must be like this #### NUMBER"
        ques_template = "You need to understand the following question and a previous answer. According to your knowledge, rethink the question and conclude the given answer is whether #### Correct or #### Wrong.\nQuestion: {}\nAnswer: {}"
        ans_template = "Rethink the question according to related knowledge. {} So the answer is #### {}."
        prompt_list = []
        for sample in data:
            cot_prompt = ques_template.format(
                sample["question_in_train"], sample["answer_in_steps_train"]
            )
            question = sample["question_in_test"]
            prompt_list.append(
                {
                    "prompt": [cot_prompt],
                    "original_question": [sample["question"]],
                    "answer": [sample["answer"]],
                    "answer_in_steps": [sample["answer_in_steps"]],
                }
            )
