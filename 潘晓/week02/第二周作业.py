import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决库冲突

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. 定义模型（用交叉熵做5分类）
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # 线性层：5维输入→5维输出（每个输出对应一个类别的的“分数”）
        self.linear = nn.Linear(input_size, num_classes)
        # 交叉熵损失（多分类专用，自带Softmax功能，直接接收线性层输出的“分数”）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 前向计算：输入→线性性层→得到5个类别的分数（无需手动加Softmax，交叉熵会处理）
        logits = self.linear(x)  # shape: (batch_size, 5)
        if y is not None:
            # 有标签时，计算交叉熵损失
            return self.loss(logits, y)
        else:
            # 无标签时，返回预测类别（取分数最大的索引）
            return torch.argmax(logits, dim=1)  # dim=1表示在5个类别中找最大值


# 2. 生成样本（5维向量，最大元素所在位置为标签）
def build_sample():
    x = np.random.random(5)  # 生成5个0~1的随机数
    max_index = np.argmax(x)  # 找到最大元素的索引（0~4）
    return x, max_index  # 输出：向量+标签（0-4）


# 3. 批量生成数据集
def build_dataset(total_num):
    X, Y = [], []
    for _ in range(total_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 多分类标签直接用整数（0-4），不用包成列表
    # 转换为PyTorch张量（特征是Float，标签是Long整数）
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 4. 评估模型准确率
def evaluate(model):
    model.eval()  # 切换到评估模式
    test_num = 1000
    x, y = build_dataset(test_num)
    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 预测类别（0-4）
    # 统计正确个数
    correct = (y_pred == y).sum().item()
    acc = correct / test_num
    print(f"测试集准确率：{acc:.4f}（{correct}/{test_num}）")
    return acc


# 5. 训练主函数
def main():
    # 配置参数
    input_size = 5  # 5维输入
    num_classes = 5  # 5分类（0-4）
    epoch_num = 30  # 训练轮数
    batch_size = 32  # 每批样本数
    train_num = 10000  # 训练样本总数
    lr = 0.01  # 学习率

    # 初始化模型、优化器
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 生成训练集
    train_x, train_y = build_dataset(train_num)

    # 记录训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式
        total_loss = 0
        # 分批次训练
        for i in range(train_num // batch_size):
            # 取一批数据
            batch_x = train_x[i*batch_size : (i+1)*batch_size]
            batch_y = train_y[i*batch_size : (i+1)*batch_size]
            # 计算损失
            loss = model(batch_x, batch_y)
            # 反向传播+更新参数
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()
        # 每轮结束后评估
        print(f"\n第{epoch+1}轮，平均损失：{total_loss/(train_num//batch_size):.6f}")
        acc = evaluate(model)
        log.append([acc, total_loss/(train_num//batch_size)])

    # 画图
    plt.plot([l[0] for l in log], label="准确率")
    plt.plot([l[1] for l in log], label="损失")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")


# 6. 用训练好的模型预测
def predict():
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load("multi_class_model.bin"))
    model.eval()

    # 测试几个样本
    test_vecs = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 最大在4→标签4
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 最大在0→标签0
        [0.2, 0.8, 0.1, 0.1, 0.1]   # 最大在1→标签1
    ]
    with torch.no_grad():
        preds = model(torch.FloatTensor(test_vecs))
    for vec, pred in zip(test_vecs, preds):
        print(f"输入：{vec}，预测最大位置：{pred.item()}")


if __name__ == "__main__":
    main()
    predict()
