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

        if (
            self.data_args.construction_mode == "n-shot"
            or self.data_args.construction_mode == "cot"
        ):
            self.train_data = self._get_data(self.data_args.train_path)
            self.test_data = self._get_data(self.data_args.test_path)
            self.data = {"train": self.train_data, "test": self.test_data}

            print("GSM8K data loaded.")
            print("Number of examples in train:", len(self.train_data))
            print("Number of examples in test:", len(self.test_data))

        # if self.split_ratio is not None:
        #     self.data = self._get_data(self.data_args.data_path)
        #     self.train_data, self.test_data = self.split(
        #         self.data, split_ratio=self.data_args.split_ratio
        #     )
        # else:
        #     self.train_data = self._get_data(self.data_args.train_path)
        #     self.test_data = self._get_data(self.data_args.test_path)
        #     if self.data_args.val_path is not None:
        #         self.val_data = self._get_data(self.data_args.val_path)

        super().__init__()

    def _get_data(self, data_path):
        retlist = []
        with open(data_path) as fp:
            lines = fp.readlines()
            for line in lines:
                json_line = json.loads(line)
                question, answer = json_line["question"], json_line["answer"]
                answer = re.sub(r"<<.*?>>", "", answer)
                result = {
                    "question": question,
                    "answer": answer,
                }
                retlist.append(result)
        return retlist

    def split(self, data, split_ratio=0.8):
        if self.data_args.split_validation:
            split_idx = int(len(data) * split_ratio)
            train_data = data[:split_idx]
            test_data = data[split_idx : split_idx + int(len(data) - split_idx) // 2]
            val_data = data[split_idx + int(len(data) - split_idx) // 2 :]
            return train_data, test_data, val_data
        else:
            split_idx = int(len(data) * split_ratio)
            train_data = data[:split_idx]
            test_data = data[split_idx:]
            return train_data, test_data


class Step2Loader(DataLoader):
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
                original_question_train = json_line["original_question_train"]
                original_question_test = json_line["original_question_test"]
                answer_in_steps_train = json_line["answer_in_steps_train"]
                answer_in_steps_test = json_line["answer_in_steps_test"]
                generated_answer = json_line["Generation Only"]

                if generated_answer == "True":
                    generated_answer = "Correct"
                elif generated_answer == "False":
                    generated_answer = "Wrong"
                else:
                    print("Error in generated answer:", generated_answer)

                # ground_truth_correctness = json_line["Correctness"]
                retlist.append(
                    {
                        "question_in_train": original_question_train,
                        "question_in_test": original_question_test,
                        "answer_in_steps_train": answer_in_steps_train,
                        "answer_in_steps_test": answer_in_steps_test,
                        "generated_answer": generated_answer,
                        # "ground_truth_correctness": ground_truth_correctness,
                    }
                )
                break
        return retlist


if __name__ == "__main__":
    data_path = "grad_std_llm/log/Mistral-7B-Instruct-v0.2/GSM8K_1-shot_0_log.jsonl"
    loader = Step2Loader(data_path)
    print(loader.data[0])
