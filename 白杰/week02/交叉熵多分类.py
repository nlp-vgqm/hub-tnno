# coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层：输入5维→输出5类
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失（内置softmax）

    def forward(self, x, y=None):
        logits = self.linear(x)  # 输出logits（未经过softmax）
        if y is not None:
            return self.loss(logits, y)  # 有标签时返回损失
        else:
            return logits  # 无标签时返回logits


# 生成样本：5维向量，标签为最大值所在维度（0-4）
def build_sample():
    x = np.random.uniform(-5, 5, size=5)  # 5维随机向量（范围扩大增加区分度）
    max_idx = np.argmax(x)  # 最大值所在维度（0-4）作为标签
    return x, max_idx


# 生成数据集（修正：先转numpy数组再转张量）
def build_dataset(total_sample_num):
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 标签为整数（0-4），无需one-hot
    # 先转为numpy数组，再转张量（解决警告）
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签用LongTensor


# 评估模型准确率
def evaluate(model):
    model.eval()
    test_num = 1000
    x, y = build_dataset(test_num)
    correct = 0
    with torch.no_grad():
        logits = model(x)  # 输出logits
        pred = torch.argmax(logits, dim=1)  # 取最大值索引作为预测类别
        correct = (pred == y).sum().item()  # 统计正确个数
    accuracy = correct / test_num
    print(f"测试集准确率：{accuracy:.4f}（{correct}/{test_num}）")
    return accuracy


def main():
    # 配置参数
    input_size = 5
    num_classes = 5  # 5个类别（0-4）
    epoch_num = 30
    batch_size = 32
    train_num = 8000
    learning_rate = 0.001

    # 初始化模型、优化器
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 生成训练数据
    train_x, train_y = build_dataset(train_num)

    # 训练记录
    log = []
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        # 按批次训练
        for batch_idx in range(train_num // batch_size):
            x = train_x[batch_idx*batch_size : (batch_idx+1)*batch_size]
            y = train_y[batch_idx*batch_size : (batch_idx+1)*batch_size]
            loss = model(x, y)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()

        # 每轮训练后评估
        avg_loss = total_loss / (train_num // batch_size)
        acc = evaluate(model)
        log.append([acc, avg_loss])
        print(f"第{epoch+1}轮，平均损失：{avg_loss:.4f}，测试准确率：{acc:.4f}")

    # 绘制训练曲线（修正：设置中文字体）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.plot([l[0] for l in log], label="准确率")
    plt.plot([l[1] for l in log], label="损失")
    plt.xlabel("训练轮次")
    plt.ylabel("数值")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")


# 预测函数（修正：列表转numpy数组后用np.round）
def predict(model_path, test_vecs):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    # 修正：添加weights_only=True消除警告
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        # 先将test_vecs转为numpy数组，再转张量
        test_vecs_np = np.array(test_vecs)
        logits = model(torch.FloatTensor(test_vecs_np))
        preds = torch.argmax(logits, dim=1)  # 预测类别
        probs = torch.softmax(logits, dim=1)  # 转换为概率（可选）
    for vec, vec_np, pred, prob in zip(test_vecs, test_vecs_np, preds, probs):
        max_val = np.max(vec_np)  # 向量中的最大值
        true_class = np.argmax(vec_np)  # 真实类别（最大值所在维度）
        # 修正：用np.round处理向量和最大值
        print(f"输入向量：{np.round(vec_np, 2)}，最大值：{np.round(max_val, 2)}（维度{true_class}）")
        print(f"预测类别：{pred.item()}，置信度：{prob[pred].item():.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
    # 测试示例（可自定义向量验证）
    test_vecs = [
        [1.2, 3.5, 2.1, 0.8, 4.2],  # 最大值在索引4→真实类别4
        [9.1, 2.3, 5.6, 3.0, 1.5],  # 最大值在索引0→真实类别0
        [0.5, 8.2, 3.3, 6.1, 2.0]   # 最大值在索引1→真实类别1
    ]
    predict("multi_class_model.bin", test_vecs)
