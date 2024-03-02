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
        sample_data = self.data[idx]
        # 将每列（每层的输出）重塑为64x64的图像，并将这些图像堆叠起来
        # 结果是一个形状为(64, 64, 64)的张量，每个通道对应于一层的输出
        sample_data_reshaped = sample_data.reshape(64, 64, 64).transpose(2, 0, 1)
        # 将numpy数组转换为torch张量
        sample_tensor = torch.tensor(sample_data_reshaped, dtype=torch.float32)
        label = self.labels[idx]
        return sample_tensor, label


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


class ModifiedCNN(nn.Module):

    def __init__(self, device, num_epochs=10, batch_size=1, lr=0.001):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 120)
        self.fc2 = nn.Linear(120, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, test_loader):
        self.train()
        for epoch in range(self.num_epochs):  # 训练10个epoch
            running_loss = 0.0
            for data, labels in train_loader:
                inputs, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            test_acc = self.test_model(test_loader)
            self.train()
            print(
                f"Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Test Acc: {test_acc}"
            )

    def test_model(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                images, labels = data.to(self.device), labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return f"{accuracy:.2f}%"


if __name__ == "__main__":
    # 实例化模型、损失函数和优化器
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    jsonl_path = "/home/kg798/grad_std_llm/log/Mistral-7B-Instruct-v0.2/Hook_GSM8K_1-shot_final_log.jsonl"  # 替换为你的 JSONL 文件路径
    batch_size = 4
    train_loader, test_loader = create_dataloaders(jsonl_path, batch_size)
    model = ModifiedCNN(device=device, num_epochs=10, lr=0.001).to(device)
    print(model)
    model.train_model(train_loader, test_loader)

    # 假设train_loader和test_loader已经定义
    # train_loader = ...
    # test_loader = ...

    # 训练和测试模型
    # train_model(train_loader)
    # test_model(test_loader)
