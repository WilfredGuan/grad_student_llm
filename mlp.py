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


class CustomDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        self.labels = []
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
        return self.data[idx], self.labels[idx]


def create_dataloaders(jsonl_path, batch_size, train_split=0.8):
    dataset = CustomDataset(jsonl_path)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    print("Train Instances:", train_size)
    print("Test Instances:", test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 定义 MLP 类
class MLP(nn.Module):
    def __init__(self, device, num_epochs=10, batch_size=1, lr=0.001):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4096 * 64, 1024)  # 第一个全连接层
        self.fc2 = nn.Linear(1024, 256)  # 第二个全连接层
        self.fc3 = nn.Linear(256, 64)  # 第三个全连接层
        self.fc4 = nn.Linear(64, 2)  # 输出层，输出大小为 2，对应二分类问题

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
            for data, labels in train_loader:
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

    def test_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return f"{accuracy:.2f}%"


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    jsonl_path = "/home/kg798/grad_std_llm/log/Mistral-7B-Instruct-v0.2/Hook_GSM8K_1-shot_final_log.jsonl"  # 替换为你的 JSONL 文件路径
    batch_size = 4
    train_loader, test_loader = create_dataloaders(jsonl_path, batch_size)
    print("Batch size:", batch_size)
    print("Train loader length:", len(train_loader))
    print("Test loader length:", len(test_loader))

    model = MLP(num_epochs=10, device=device, batch_size=batch_size, lr=0.0015).to(
        device
    )
    print(model)
    model.train_model(train_loader=train_loader, test_loader=test_loader)
    print("Finished Training")
    model.test_model(test_loader=test_loader)
