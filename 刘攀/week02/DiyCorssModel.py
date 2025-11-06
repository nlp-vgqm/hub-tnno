# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import matplotlib

matplotlib.use('TkAgg')  # 切换后端
import matplotlib.pyplot as plt
import numpy as np

"""
基于pytorch框架实现五维数组分类任务
规律：x是5维向量，若第i个元素最大（i=0,1,2,3,4），则标签为i（0对应第一类）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5维（对应5个类别）
        self.loss = nn.CrossEntropyLoss()  # 多分类交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5) -> 输出5类别的logits
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def build_sample():
    # 生成五维随机数组（正态分布）
    X = torch.randn(5)  # 形状：(n_samples, 5)
    # 计算每个样本的最大值索引（标签）：若第一个元素最大，标签为0（第一类）函数argmax返回向量那个最大
    y = np.zeros(5)
    y[torch.argmax(X)] =1
    return X, y
# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    # 步骤1：将列表转为统一的numpy数组
    X_np = np.array(X)  # 转为二维numpy数组，形状：(total_sample_num, 5)
    Y_np = np.array(Y)  # 转为一维numpy数组，形状：(total_sample_num,)
    # 步骤2：将numpy数组转为PyTorch张量（推荐用 torch.from_numpy，更高效）
    return torch.FloatTensor(X_np), torch.FloatTensor(Y_np)
# 测试代码
# 用来测试每轮模型的准确率
# 评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_t[np.argmax(y_p)] == 1:
                correct += 1    # 计算返回的5维向量里最大值与真实向量值里的一致，则表示正确
            else:
                wrong += 1
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
    # 生成训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 按批次训练
        for batch_idx in range(train_sample // batch_size):
            # 截取批次数据
            x = train_x[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y = train_y[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # 计算损失
            loss = model(x, y)
            print("main:", loss)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "diyCorssmodel.bin")
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
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)