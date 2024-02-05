import numpy as np
import json
import re


class DataLoader:
    def __init__(self):
        self.n = len(self.data)

    def get_data(self):
        return self.data

    def __getitem__(self, idx):
        return self.data[idx]

    def __next__(self):
        return self.__iter__().__next__()

    def __len__(self):
        return self.n


class GSM8KLoader(DataLoader):
    def __init__(self, data_args):
        self.data_args = data_args
        if self.split_ratio is not None:
            self.data = self._get_data(self.data_args.data_path)
            self.train_data, self.test_data = self.split(
                self.data, split_ratio=self.data_args.split_ratio
            )
        else:
            self.train_data = self._get_data(self.data_args.train_path)
            self.test_data = self._get_data(self.data_args.test_path)
            if self.data_args.val_path is not None:
                self.val_data = self._get_data(self.data_args.val_path)

        super().__init__()

    def _get_data(self, data_path):
        retlist = []
        with open(data_path) as fp:
            lines = fp.readlines()
            for line in lines:
                json_line = json.loads(line)
                question, answer = json_line["question"], json_line["answer"]
                answer = re.sub(r"<<.*?>>", "", answer)
                retlist.append({"question": question, "answer": answer})
        print("GSM8K data loaded.")
        print("Number of examples:", len(retlist))
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


if __name__ == "__main__":
    data_path = "grad_std_llm/data/GSM8K/test.jsonl"
    loader = GSM8KLoader(data_path)
    print(loader.data[0])
