# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
def set_seed(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)          # Python内置随机数
    np.random.seed(seed)       # NumPy随机数
    torch.manual_seed(seed)    # PyTorch CPU随机数
    torch.cuda.manual_seed(seed) # PyTorch GPU随机数
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    # 确保卷积操作也是确定性的（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

# 在程序开始处调用
set_seed(10)  # 你可以选择任何喜欢的数字作为种子

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss() # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        # y_pred = self.softmax(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律

def build_sample():
    x = np.random.random(5)
    if x[0] > x[1] and x[0] > x[2] and x[0] > x[3] and x[0] > x[4]:
        return x, 0
    elif x[1] > x[0] and x[1] > x[2] and x[1] > x[3] and x[1] > x[4]:
        return x, 1
    elif x[2] > x[0] and x[2] > x[1] and x[2] > x[3] and x[2] > x[4]:
        return x, 2
    elif x[3] > x[0] and x[3] > x[1] and x[3] > x[2] and x[3] > x[4]:
        return x, 3
    elif x[4] > x[0] and x[4] > x[1] and x[4] > x[2] and x[4] > x[3]:
        return x, 4


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 得到概率分布，形状: (100, 5)
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        true_classes = y.squeeze().long()  # 确保标签形状匹配

        correct = (predicted_classes == true_classes).sum().item()
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "../py312/model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        predicted_classes = torch.argmax(result, dim=1) + 1  # 获取预测类别
    for vec, pred_class, res in zip(input_vec, predicted_classes, result):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, pred_class.item(), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.9724256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.73625847,0.34675372,0.19871392],
                [0.29349776,0.59416669,0.12579291,0.71567412,0.1358894],
                [0.29349776,0.59416669,0.12579291,0.71567412,0.8358894]]
    predict("../py312/model.bin", test_vec)
