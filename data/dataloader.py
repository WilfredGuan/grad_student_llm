import numpy as np
import json
import re


class DataLoader:
    def __init__(self):
        self.n = len(self.data)

    def get_data(self):
        return self.data

    def __next__(self):
        return self.__iter__().__next__()

    def __len__(self):
        return self.n


class GSM8KLoader(DataLoader):
    def __init__(self, data_args):
        self.data_args = data_args
        if self.data_args.construction_mode == "zero-shot":
            self.data = self._get_data(self.data_args.test_path)

        if self.data_args.construction_mode == "n-shot":
            self.train_data = self._get_data(self.data_args.train_path)
            self.test_data = self._get_data(self.data_args.test_path)
            self.data = {"train": self.train_data, "test": self.test_data}

            print("GSM8K data loaded.")
            print("Number of examples in train:", len(self.train_data))
            print("Number of examples in test:", len(self.test_data))

        super().__init__()

    def _get_data(self, data_path):
        retlist = []
        with open(data_path) as fp:
            lines = fp.readlines()
            for line in lines:
                json_line = json.loads(line)
                question, answer = json_line["question"], json_line["answer"]
                # clean up the <<>> annotations in the dataset
                answer = re.sub(r"<<.*?>>", "", answer)
                result = {
                    "question": question,
                    "answer": answer,
                }
                retlist.append(result)
        return retlist


class Step2KnowledgeLoader(DataLoader):
    def __init__(self, data_args):
        self.data_args = data_args
        self.data = self._get_data(self.data_args.data_path)
        super().__init__()

    def _get_data(self, data_path):
        retlist = []
        with open(data_path) as fp:
            lines = fp.readlines()
            for line in lines:
                json_line = json.loads(line)
                ques = json_line["original_question_test"]
                ans = json_line["answer_in_steps_test"]
                generated_ans = json_line["Generation Only"]
                ground_truth_correctness = json_line["Correctness"]

                if ground_truth_correctness == "True":
                    ground_truth_correctness = "Correct"
                elif ground_truth_correctness == "False":
                    ground_truth_correctness = "Wrong"
                else:
                    print("Error in generated answer:", ground_truth_correctness)

                retlist.append(
                    {
                        "ques": ques,
                        "ans": ans,
                        "generated_ans": generated_ans,
                        "ground_truth_correctness": ground_truth_correctness,
                    }
                )
        return retlist


if __name__ == "__main__":
    data_path = (
        "/home/kg798/grad_std_llm/data/step1_1-shot/GSM8K_1-shot_final_log.jsonl"
    )
    loader = Step2KnowledgeLoader(data_path)
    print(loader.data[0])
