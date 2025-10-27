# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务：五维随机向量，最大的数字在哪维就属于哪一类,使用交叉熵损失函数
规律：五维随机向量，最大的数字在哪维就属于哪一类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 线性层，输入5维，输出5维（对应5个类别）
        self.linear = nn.Linear(input_size, num_classes)
        # 使用交叉熵损失函数（包含softmax，所以不需要额外添加激活函数）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 线性层输出 (batch_size, 5) -> (batch_size, 5)
        y_pred = self.linear(x)
        if y is not None:
            # 预测值和真实值计算损失，计算交叉熵损失（y应该是类别索引，不是one-hot编码）
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成一个样本：五维随机向量，最大值所在维度为类别标签
def build_sample():
    x = np.random.random(5)  # 生成5维随机向量
    y = np.argmax(x)  # 最大值所在的索引作为类别（0-4）
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意Y是LongTensor


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计各类别样本数量
    class_count = [0] * 5
    for label in y:
        class_count[label] += 1
    print("各类别样本数量:", class_count)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 获取预测类别（取最大值的索引）
        predicted_classes = torch.argmax(y_pred, dim=1)

        for y_p, y_t in zip(predicted_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 错误预测个数：%d, 正确率：%f" % (correct, wrong, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 5000  # 每轮训练样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)

    # 选择优化器（使用Adam优化器）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    print("开始训练...")
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 批次训练
        for batch_index in range(train_sample // batch_size):
            # 获取当前批次数据
            start_idx = batch_index * batch_size
            end_idx = (batch_index + 1) * batch_size
            x = train_x[start_idx:end_idx]
            y = train_y[start_idx:end_idx]

            # 前向传播计算损失
            loss = model(x, y)

            # 反向传播
            loss.backward()

            # 更新权重
            optim.step()

            # 梯度清零
            optim.zero_grad()

            watch_loss.append(loss.item())

        # 计算平均损失
        avg_loss = np.mean(watch_loss)

        # 评估模型
        acc = evaluate(model)

        print("第%d轮训练 - 平均损失: %f, 准确率: %f" % (epoch + 1, avg_loss, acc))
        log.append([acc, avg_loss])

        # 如果准确率达到95%以上，提前结束训练
        if acc > 0.95:
            print("训练提前结束，准确率已达到95%以上")
            break

    # 保存模型
    torch.save(model.state_dict(), "multiclass_model.bin")
    print("模型已保存为 multiclass_model.bin")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], 'b-', label="准确率")
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.title('训练准确率曲线')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], 'r-', label="损失值")
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model


# 使用训练好的模型进行预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        # 模型预测
        results = model.forward(torch.FloatTensor(input_vec))
        # 获取预测概率（使用softmax）
        probabilities = torch.softmax(results, dim=1)
        # 获取预测类别
        predicted_classes = torch.argmax(results, dim=1)

        for i, (vec, pred_class, prob) in enumerate(zip(input_vec, predicted_classes, probabilities)):
            print("样本%d: 输入向量 %s" % (i + 1, [round(x, 3) for x in vec]))
            print("      真实类别: %d (最大值在索引%d)" % (np.argmax(vec), np.argmax(vec)))
            print("      预测类别: %d" % pred_class.item())
            print("      各类别概率: %s" % [round(p, 3) for p in prob.numpy()])
            print("      预测结果: %s" % ("正确" if pred_class.item() == np.argmax(vec) else "错误"))
            print("-" * 50)


if __name__ == "__main__":
    # 训练模型
    trained_model = main()

    # 测试预测功能
    print("\n" + "=" * 60)
    print("模型预测测试:")
    print("=" * 60)

    # 生成几个测试样本
    test_samples = []
    for i in range(5):
        x, y = build_sample()
        test_samples.append(x)

    predict("multiclass_model.bin", test_samples)
