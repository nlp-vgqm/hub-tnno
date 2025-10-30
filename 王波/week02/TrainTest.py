# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维随机向量中最大的数字在哪维就属于哪一类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        # 线性层：输入5维，输出5维（对应5个类别）
        self.linear = nn.Linear(input_size, num_classes)
        # 使用交叉熵损失，它内部包含Softmax，所以不需要在模型中显式添加
        # 交叉熵：nn.CrossEntropyLoss()：它结合了 LogSoftmax 和 NLLLoss 两个操作
        # 该函数直接接受模型的原始输出（logits）和目标类别标签，无需在模型输出层额外添加 softmax 激活函数

        # 自定义初始化，使用较小的权重
        nn.init.normal_(self.linear.weight, mean=1, std=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, y=None):
        # 直接输出线性层的结果（logits）
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)

        if y is not None:
            # 使用交叉熵损失函数
            # 注意：y需要是长整型，并且形状为(batch_size,)
            loss = nn.CrossEntropyLoss()(y_pred, y.squeeze().long())
            return loss
        else:
            # 预测时返回概率分布,Softmax:将每个类别的输出分数转换为概率，使得所有类别的概率之和为1。
            # y_pred 是线性层的输出，形状为 (batch_size, 5)
            return nn.Softmax(dim=1)(y_pred)


# 造数据
def build_sample_2():
    # _s = np.random.randint(0, 1000)
    # if _s < 50 and _s % 2 == 0:
    #     x = np.random.randint(_s, _s + 5, 5)
    # elif _s < 50 and _s % 2 == 1:
    #     x = np.arange(_s + 5, _s, -1)
    # elif _s >= 950 and _s % 2 == 0:
    #     x = np.arange(_s + 5, _s, -1)
    # elif _s >= 950 and _s % 2 == 1:
    #     x = np.random.randint(_s, _s + 8, 5)
    # elif 450 <= _s < 500 and _s % 2 == 0:
    #     x = np.random.randint(_s - 5, _s, 5)
    # elif 500 <= _s < 550 and _s % 2 == 1:
    #     x = np.random.randint(_s, _s + 10, 5)
    # else:
    #     x = np.random.randint(50, 951, 5)
    x = np.random.randint(0, 1001, 5)
    y = np.argmax(x)  # 最大值所在的维度索引(0-4)

    return x, y


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample_2()
        X.append(x)
        Y.append([y])  # 保持形状一致
    # 转换为PyTorch张量
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % len(x))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        # 获取预测类别（概率最大的索引）
        _, predicted = torch.max(y_pred.data, 1)  # torch.max：获取张量中最大的值(第一个返回值：指定维度上的最大值,第二个返回值：最大值所在的索引)
        for y_p, y_t in zip(predicted, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 输出类别数
    learning_rate = 0.0005  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零

            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_2.bin")

    # 画图  # 第一轮损失值较大，影响图标显示精度，去掉
    plt.plot(range(1, len(log)), [l[0] for l in log[1:]], label="acc")
    plt.plot(range(1, len(log)), [l[1] for l in log[1:]], label="loss")
    # 添加坐标轴标签和标题
    plt.xlabel("Epoch")  # 横坐标名称
    plt.ylabel("Value")  # 纵坐标名称
    plt.title("Training Process")  # 图表标题
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("加载模型完成")
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，返回概率分布
        # 获取预测类别和对应的概率
        probabilities, predicted = torch.max(result, 1)

    for vec, pred, prob in zip(input_vec, predicted, probabilities):
        true_idx = np.argmax(vec)
        print(
            f"输入：{vec}, 真实最大值索引：{true_idx}, 预测索引：{pred.item()}, 概率：{prob.item():.4f}, 是否正确：{true_idx == pred.item()}")


if __name__ == "__main__":
    main()
    test_vec = [
        [744, 867, 910, 702, 94],
        [525, 787, 602, 959, 786],
        [298, 736, 211, 595, 986],
        [307, 551, 927, 299, 746],
        [977, 590, 356, 832, 541],
        [719, 535, 983, 52, 550],
        [58, 942, 709, 920, 579],
        [463, 168, 205, 514, 857],
        [87, 577, 404, 407, 448],
        [137, 830, 31, 633, 727],
        [100, 101, 102, 103, 104],
        [999, 998, 997, 996, 995],
        [500, 501, 504, 503, 502]]
    predict("model_2.bin", test_vec)
