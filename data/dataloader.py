import numpy as np
import json
import re


class DataLoader:
    def __init__(self, data, data_name, data_path):
        self.data = data
        self.data_name = data_name
        self.data_path = data_path
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
    def __init__(self, data_path, train_ratio=0.8, validation=False):
        super().__init__(
            data=self._get_data(data_path),
            data_name="GSM8K",
            data_path=data_path,
        )

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

    def split(self, data, split_ratio=0.8, validation=False):
        if validation:
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
