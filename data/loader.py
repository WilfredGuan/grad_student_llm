import numpy as np
import json
import re


class Loader:
    def __init__(self, data, data_name, data_path, batch_size):
        self.data = data
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size

        self.n = len(self.data)
        self.n_batches = int(np.ceil(self.n / self.batch_size))
        self.indices = np.arange(self.n)

    def __iter__(self):
        np.random.shuffle(self.indices)

        for i in range(self.n_batches):
            batch_indices = self.indices[
                i * self.batch_size : (i + 1) * self.batch_size
            ]
            # print(batch_indices)
            batch = [self.data[index] for index in batch_indices]
            yield batch

    def get_data(self):
        return self.data

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        return self.data[idx]

    def __next__(self):
        return self.__iter__().__next__()


class GSM8KLoader(Loader):
    def __init__(self, data_path, batch_size):
        super().__init__(
            data=self.get_data(data_path),
            data_name="GSM8K",
            data_path=data_path,
            batch_size=batch_size,
        )

    def get_data(self, data_path):
        retlist = []
        with open(data_path) as fp:
            lines = fp.readlines()
            for line in lines:
                json_line = json.loads(line)
                question, answer = json_line["question"], json_line["answer"]
                answer = re.sub(r"<<.*?>>", "", answer)
                retlist.append({"question": question, "answer": answer})
        # print("GSM8K data loaded.")
        # print("Number of examples:", len(retlist))
        # print("Example:", retlist[0])
        return retlist


if __name__ == "__main__":
    data_path = "models/grad_std_llm/data/GSM8K/test.jsonl"
    loader = GSM8KLoader(data_path, 1)
    for batch in loader:
        print(batch)
        break
