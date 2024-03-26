from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, step1_data_path, step2_data_path, step2_state_path):
        self.data = []
        self.labels = []
        self.original_data = []
        self.error_idx = []
        self.export_path = "/home/kg798/grad_std_llm/data/step3_1-shot+knowledge/step3_1-shot+knowledge.jsonl"
        self.step1_data = self.load_data(step1_data_path, "step1")
        print("step1_data loaded, lenth=", len(self.step1_data))
        self.step2_data = self.load_data(step2_data_path, "step2")
        print("step2_data loaded, lenth=", len(self.step2_data))
        self.step2_state = self.load_data(step2_state_path, "step2_state")
        print("step2_state loaded, lenth=", len(self.step2_state))
        print("Processing data...")
        self.process_and_export()
        print("Data processed and exported to", self.export_path)

    def load_data(self, jsonl_path, name):
        data = []
        with open(jsonl_path, "r") as file:
            for line in tqdm(file, desc=f"Loading {name}"):
                data_item = json.loads(line)
                data.append(data_item)
        return data

    def process_and_export(self):
        for idx in tqdm(range(len(self.step1_data))):
            step1_data = self.step1_data[idx]
            step2_data = self.step2_data[idx]
            step2_state = self.step2_state[idx]
            combined_data = np.empty((4096, 64), dtype=np.float32)

            if (
                step1_data["Correctness"] != "True"
                and step1_data["Correctness"] != "False"
            ):
                self.error_idx.append({"file": "step1", "idx": idx})
                continue
            if (
                step2_data["Cleaned Generation"] != "Correct"
                and step2_data["Cleaned Generation"] != "Wrong"
            ):
                self.error_idx.append({"file": "step2", "idx": idx})
                continue

            for i in range(32):
                self_attn_data = np.array(
                    step2_state["outputs"]["last"][f"model.layers.{i}.self_attn.o_proj"]
                ).reshape(4096)
                mlp_data = np.array(
                    step2_state["outputs"]["last"][f"model.layers.{i}.mlp"]
                ).reshape(4096)
                # 按顺序合并到 combined_data 数组
                combined_data[:, 2 * i] = self_attn_data
                combined_data[:, 2 * i + 1] = mlp_data
            self.data.append(combined_data)

            if (
                step1_data["Correctness"] == "True"
                and step2_data["Cleaned Generation"] == "Correct"
            ):
                label = 1
            elif (
                step1_data["Correctness"] == "False"
                and step2_data["Cleaned Generation"] == "Wrong"
            ):
                label = 0
            else:
                label = 2

            self.labels.append(label)
            self.original_data.append({"step1": step1_data, "step2": step2_data})

        with open(self.export_path, "w") as file:
            for idx in tqdm(range(len(self)), desc="Writing to file"):
                # 获取单个数据项
                data_item, label_item = self.__getitem__(idx)
                json_str = json.dumps(
                    {
                        "data": data_item.tolist(),
                        "label": label_item,
                        "original_data": self.original_data[idx],
                    }
                )
                file.write(json_str + "\n")
        # print("error_idx", self.error_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    step1_data_path = (
        "/home/kg798/grad_std_llm/data/step1_1-shot/GSM8K_1-shot_final_log.jsonl"
    )
    step2_data_path = "/home/kg798/grad_std_llm/data/step2_knowledge/GSM8K_step2knowledge-zero-shot_final_log.jsonl"
    step2_state_path = "/home/kg798/grad_std_llm/data/step2_knowledge/Hook_GSM8K_step2knowledge-zero-shot_final_log.jsonl"
    data = CustomDataset(step1_data_path, step2_data_path, step2_state_path)
