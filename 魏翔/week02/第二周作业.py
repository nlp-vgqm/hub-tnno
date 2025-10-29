import numpy.random
import torch
import random
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def check(x):
    t = 0
    max_idx = 0
    for idx, x in enumerate(x):
        if x > t:
            t = x
            max_idx = idx
    return max_idx


class MyTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyTorch, self).__init__()
        self.f1 = nn.Linear(input_size, 100)
        self.activate1 = nn.ReLU()
        self.f3 = nn.Linear(100, output_size)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), 0.001)
        self.to(device)

    def forward(self, x, y=None):
        x = self.f1(x)
        x = self.activate1(x)
        x = self.f3(x)
        if y is None:
            return x
        else:
            return self.loss(x, y)


# def f(x):
#     a, b, c = 3, 15, 3
#     return a * x ** 2 + b * x + c
#
#
# def build_sample():
#     x = random.random()
#     return x, f(x)
#
#
# def build_dateset(total_sample_number):
#     X = []
#     Y = []
#     for i in range(total_sample_number):
#         x, y = build_sample()
#         X.append([x])
#         Y.append([y])
#     return torch.FloatTensor(X), torch.FloatTensor(Y)
#

class SimpleDataset(Dataset):
    def __init__(self, num_sample=1000):
        self.num_sample = num_sample

        self.X = []
        self.Y = []
        self._generate_date()

    def _generate_date(self):
        for i in range(self.num_sample):
            x = np.random.rand(5)
            self.X.append(x)
            self.Y.append(check(x))
        self.X = torch.FloatTensor(self.X)
        self.Y = torch.LongTensor(self.Y)
        print(self.X, self.Y)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


dataset = SimpleDataset()
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = int(0.15 * len(dataset))

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(len(train_dataset), len(val_dataset), len(test_dataset))
dataloader1 = DataLoader(train_dataset, 10, True)
dataloader2 = DataLoader(test_dataset, 1, True)
model = MyTorch(5, 5)

epochs = 200

for epoch in range(epochs):
    model.train()
    for index, (x, y) in enumerate(dataloader1):
        x, y = x.to(device), y.to(device)
        # print(f"index: {index}, input:{x}, output:{y}")
        loss = model(x, y)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        # print(f"epoch:{epoch}, loss:{loss.item()}")


cnt = 0
with torch.no_grad():
        model.eval()
        for index, (x, y) in enumerate(dataloader2):
            x, y = x.to(device), y.to(device)
            # print(f"index: {index}, input:{x}, output:{y}")
            # print(torch.max(model(x), 1)[1])
            if torch.max(model(x), 1)[1] == y:
                cnt += 1

print(cnt)
print(len(dataloader2))
print("预测准确率：", cnt / len(dataloader2) * 100, "%")
print(torch.cuda.is_available())
