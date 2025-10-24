作业内容：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类


# coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
5分类任务：
给一个随机的5维向量，判断最大值所在的维度（0~4），作为类别标签。
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, x, y=None):
        logits = self.linear(x)  # 原始分数
        if y is not None:
            return self.loss(logits, y)  # y: LongTensor, shape (batch_size,)
        else:
            return torch.softmax(logits, dim=1)  # 推理时返回概率分布


# 生成一个样本
def build_sample():
    x = np.random.random(5).astype(np.float32)  # 保证是 float32
    y = np.argmax(x).astype(np.int64)           # 最大值索引，int64
    return x, y

# 生成数据集（优化版：一次性转换为 NumPy 数组，避免警告）
def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X, dtype=np.float32)  # 一次性转换
    Y = np.array(Y, dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(Y)


# 评估模型
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        y_pred = model(x)
        pred_classes = torch.argmax(y_pred, dim=1)
        correct = (pred_classes == y).sum().item()
    acc = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    return acc


# 训练主函数
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5
    learning_rate = 0.001

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch+1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model_multi.bin")

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    pred_classes = torch.argmax(result, dim=1)
    for vec, cls, probs in zip(input_vec, pred_classes, result):
        print(f"输入：{vec}, 预测类别：{cls.item()}, 概率分布：{probs.numpy()}")


if __name__ == "__main__":
    main()
    test_vec = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.9, 0.1, 0.2, 0.3, 0.4],
        [0.2, 0.8, 0.1, 0.05, 0.05]
    ]
    predict("model_multi.bin", test_vec)
