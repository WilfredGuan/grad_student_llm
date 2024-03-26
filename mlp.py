import torch
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os


class CustomDataset(Dataset):
    def __init__(self, jsonl_path, mode):
        self.data = []
        self.labels = []
        self.original_data = []
        if mode == "step3":
            with open(jsonl_path, "r") as file:
                for line in tqdm(file, desc="Loading Data"):
                    # print(line)
                    json_obj = json.loads(line.strip())
                    self.data.append(np.array(json_obj["data"], dtype=np.float32))
                    self.labels.append(json_obj["label"])
                    self.original_data.append(json_obj["original_data"])
        else:
            with open(jsonl_path, "r") as file:
                for line in file:
                    json_obj = json.loads(line.strip())
                    # 初始化一个空的数组来存储合并后的数据
                    combined_data = np.empty((4096, 64), dtype=np.float32)
                    for i in range(32):
                        # 提取 self_attn 和 mlp 层的数据
                        self_attn_data = np.array(
                            json_obj["outputs"]["last"][
                                f"model.layers.{i}.self_attn.o_proj"
                            ]
                        ).reshape(4096)
                        mlp_data = np.array(
                            json_obj["outputs"]["last"][f"model.layers.{i}.mlp"]
                        ).reshape(4096)
                        # 按顺序合并到 combined_data 数组
                        combined_data[:, 2 * i] = self_attn_data
                        combined_data[:, 2 * i + 1] = mlp_data
                    self.data.append(combined_data)
                    label = 1 if json_obj["result"] == "True" else 0
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.original_data[idx]


def create_dataloaders(jsonl_path, batch_size, mode, train_split=0.8):
    if train_split != 0:
        dataset = CustomDataset(jsonl_path, mode)
        train_size = int(train_split * len(dataset))
        test_size = len(dataset) - train_size
        print("Train Instances:", train_size)
        print("Test Instances:", test_size)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        dataset = CustomDataset(jsonl_path, mode)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader


# 定义 MLP 类
class MLP(nn.Module):
    def __init__(self, device, num_epochs=10, batch_size=1, lr=0.001):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096 * 64, 1024)  # 第一个全连接层
        self.fc2 = nn.Linear(1024, 256)  # 第二个全连接层
        self.fc3 = nn.Linear(256, 64)  # 第三个全连接层
        self.fc4 = nn.Linear(64, 3)  # 输出层，输出大小为 2，对应二分类问题

        self.relu = nn.ReLU()  # 激活函数

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self, train_loader, test_loader):
        for epoch in range(self.num_epochs):
            total_loss = 0
            for data, labels, _ in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            test_acc = self.test_model(test_loader)
            self.train()
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}, Test Acc: {test_acc}"
            )

    def test_model(
        self,
        test_loader,
        file_path="3->1_misclassified_samples.jsonl",
        test_export=False,
    ):
        self.eval()
        correct = 0
        total = 0
        if test_export:
            misclassified_samples = []
        with torch.no_grad():
            for data, labels, original_data in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if test_export:
                    misclassified = predicted.item() != labels.item()
                    # 如果有预测错误的数据
                    if misclassified:
                        misclassified_samples.append(
                            {
                                "predicted": predicted.item(),
                                "actual": labels.item(),
                                "data": original_data,
                            }
                        )
        if test_export:
            with open(file_path, "w") as f:
                for item in misclassified_samples:
                    f.write(json.dumps(item) + "\n")

        accuracy = 100 * correct / total
        return f"{accuracy:.2f}%"

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.to(self.device)


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    jsonl_path = "/home/kg798/grad_std_llm/data/step1_1-shot/step1_1-shot_dataset.jsonl"  # 替换为你的 JSONL 文件路径
    batch_size = 1
    mode = "step3"

    flag = "train"

    if flag == "train":
        train_loader, test_loader = create_dataloaders(
            jsonl_path, batch_size, mode, train_split=0.8
        )
        model = MLP(num_epochs=10, device=device, batch_size=batch_size, lr=0.001).to(
            device
        )
        model.train_model(train_loader=train_loader, test_loader=test_loader)
        print("Finished Training")
        print("Testing...")
        print(model.test_model(test_loader=test_loader))
        print("Saving Model...")
        model.save_model(os.getcwd() + "/mlp_save/step_1_model.pth")

    if flag == "test":
        test_loader = create_dataloaders(jsonl_path, batch_size, mode, train_split=0)
        model = MLP(num_epochs=10, device=device, batch_size=batch_size, lr=0.001).to(
            device
        )
        model.load_model(os.getcwd() + "/mlp_save/model.pth")
        print(model.test_model(test_loader=test_loader, test_export=True))
