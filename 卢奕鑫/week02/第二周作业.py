# coding:utf8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
多分类任务：5维随机向量中，最大值所在的维度即为类别（0-4）
使用交叉熵损失函数（CrossEntropyLoss）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 线性层：输入5维，输出5类（对应0-4类的得分）
        self.linear = nn.Linear(input_size, num_classes)
        # 交叉熵损失包含了Softmax和NLLLoss，这里不需要手动加Softmax
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 计算类别得分（logits），形状为(batch_size, 5)
        logits = self.linear(x)
        if y is not None:
            # 若提供标签，计算交叉熵损失
            return self.loss(logits, y)
        else:
            # 若不提供标签，返回预测类别（概率最大的类别）
            return torch.argmax(logits, dim=1)


# 生成样本：5维随机向量，最大值所在维度为类别（0-4）
def build_sample():
    x = np.random.random(5)  # 生成5个0-1之间的随机数
    max_idx = np.argmax(x)  # 找到最大值的索引（0-4），作为类别
    return x, max_idx


# 生成数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 交叉熵损失的标签是类别索引（无需One-Hot）
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签需为LongTensor类型


# 评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型返回预测类别（0-4）
        correct = (y_pred == y).sum().item()  # 计算正确个数
    acc = correct / test_sample_num
    print(f"测试集准确率：{correct}/{test_sample_num} = {acc:.4f}")
    return acc


def main():
    # 配置参数
    input_size = 5  # 输入维度
    num_classes = 5  # 类别数（0-4）
    epoch_num = 3000  # 训练轮数
    batch_size = 200  # 批次大小
    train_sample = 10000  # 训练样本总数
    learning_rate = 0.01  # 学习率

    # 初始化模型、优化器
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 生成训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练记录
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        # 按批次训练
        for batch_idx in range(train_sample // batch_size):
            # 截取批次数据
            batch_x = train_x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_y = train_y[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # 计算损失
            loss = model(batch_x, batch_y)
            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            total_loss += loss.item()

        # 每轮结束后评估
        print(f"\n第{epoch+1}轮，平均损失：{total_loss / (train_sample//batch_size):.6f}")
        acc = evaluate(model)
        log.append([acc, total_loss / (train_sample//batch_size)])

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")
    print("\n模型权重：")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.numpy()}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([l[0] for l in log], label="Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([l[1] for l in log], label="Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 预测函数
def predict(model_path, input_vecs):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(input_vecs)
        y_pred = model(x)  # 预测类别

    for vec, pred in zip(input_vecs, y_pred):
        max_val = np.max(vec)
        true_class = np.argmax(vec)  # 真实类别（最大值所在维度）
        print(f"输入向量：{vec}")
        print(f"最大值：{max_val}，真实类别：{true_class}，预测类别：{pred.item()}\n")


if __name__ == "__main__":
    main()
    # 测试预测
    test_vecs = [
        [0.1, 0.51, 0.5, 0.2, 0.4],  # 最大值在索引2（类别2）
        [0.9, 0.1, 0.2, 0.39, 0.4],  # 最大值在索引0（类别0）
        [0.2, 0.503, 0.3, 0.1, 0.5],  # 最大值在索引1（类别1）
        [0.5008, 0.3, 0.2, 0.5007, 0.1],  # 最大值在索引3（类别3）
        [0.3, 0.2, 0.79999, 0.1, 0.8]   # 最大值在索引4（类别4）
    ]
    predict("multiclass_model.bin", test_vecs)
