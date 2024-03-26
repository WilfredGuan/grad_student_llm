from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, step1_data_path, step1_state_path):
        self.data = []
        self.labels = []
        self.original_data = []
        self.error_idx = []
        self.export_path = (
            "/home/kg798/grad_std_llm/data/step1_1-shot/step1_1-shot_dataset.jsonl"
        )
        self.step1_data = self.load_data(step1_data_path, "step1")
        print("step1_data loaded, lenth=", len(self.step1_data))
        self.step1_state = self.load_data(step1_state_path, "step1_state")
        print("step1_state loaded, lenth=", len(self.step1_state))
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
            step1_state = self.step1_state[idx]
            combined_data = np.empty((4096, 64), dtype=np.float32)

            for i in range(32):
                self_attn_data = np.array(
                    step1_state["outputs"]["last"][f"model.layers.{i}.self_attn.o_proj"]
                ).reshape(4096)
                mlp_data = np.array(
                    step1_state["outputs"]["last"][f"model.layers.{i}.mlp"]
                ).reshape(4096)
                combined_data[:, 2 * i] = self_attn_data
                combined_data[:, 2 * i + 1] = mlp_data
            self.data.append(combined_data)

            if step1_data["Correctness"] == "True":
                label = 1
            elif step1_data["Correctness"] == "False":
                label = 0

            self.labels.append(label)
            self.original_data.append(step1_data)

        with open(self.export_path, "w") as file:
            for idx in tqdm(range(len(self)), desc="Writing to file"):
                data_item, label_item, original_data_item = self.__getitem__(idx)
                json_str = json.dumps(
                    {
                        "data": data_item.tolist(),
                        "label": label_item,
                        "original_data": original_data_item,
                    }
                )
                file.write(json_str + "\n")
        # print("error_idx", self.error_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.original_data[idx]


if __name__ == "__main__":
    step1_data_path = (
        "/home/kg798/grad_std_llm/data/step1_1-shot/GSM8K_1-shot_final_log.jsonl"
    )
    step1_state_path = (
        "/home/kg798/grad_std_llm/data/step1_1-shot/Hook_GSM8K_1-shot_final_log.jsonl"
    )
    data = CustomDataset(step1_data_path, step1_state_path)
