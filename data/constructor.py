import json
import re


class Constructor:
    def __init__(self, data, *kwargs):
        self.data = data

    def get_data(self):
        pass

    def _get_synthetic_data(self):
        pass


class GSM8KConstructor(Constructor):
    def __init__(self, data_args, *kwargs):
        super().__init__(data=data)

        self.data = self._construct(data)

    def _construct(self, data):
        print("Constructing GSM8K data...")
        # template = "You are a math teacher thinking step by step. Your question is: {} \n\n Your final answer must be like this #### NUMBER"
        template = "You are a math teacher thinking step by step. Replace your numeric answers to YOUR_NUMBER in this format #### YOUR_NUMBER.\n\nYour question is: {}\n\n Your answer is:"
        prompt_list = []
        for sample in data:
            prompt = template.format(sample["question"])
            prompt_list.append({"prompt": prompt, "answer": sample["answer"]})
        # print("GSM8K data constructed.")
        # print("Number of examples:", len(prompt_list))
        # print("Example:", prompt_list[0])
        return prompt_list

    def _cot_construct(self, data):
        pass
