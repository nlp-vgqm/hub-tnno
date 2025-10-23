# coding:utf8

# 解决 OpenMP 库冲突问题
import os

from sqlalchemy.util import ellipses_string

# 在导入torch之前设置
os.environ['KMP_DYNAMIC'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，返回当前向量中最大值的维度

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.softmax  # CrossEntropyLoss内部模式使用了softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # loss函数使用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 预测值和真实值计算损失
        else:
            return torch.softmax(x,dim=1)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，返回最大值对应的维度
def build_sample():
    x = np.random.random(5)
    return x,np.argmax(x)

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.IntTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计每个类别的样本数量
    class_counts = [0] * 5
    for label in y:
        class_counts[int(label)] += 1
    print(f"本次预测集中各类别样本数量:{class_counts}", )

    correct, wrong = 0, 0
    with torch.no_grad():  # 关闭梯度计算
        y_pred = model(x)  # 模型预测，返回形状为(100, 5)的概率分布
        # print(f"y_pred:{y_pred}")
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别索引
        # print(f"true_classes:{true_classes}")
        for y_p, y_t in zip(predicted_classes, y):
            if y_p == y_t:  # 预测类别与真实类别相同
                correct += 1
            else:
                wrong += 1

    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong):.2%}")
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
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
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):6f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
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
    # for name,param in model.state_dict().items():
    #     print(f"{name}:{param.shape}")
    print("模型加载成功")

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测得到概率分布
    #处理预测结果
    error,count=0,0
    for vec, res in zip(input_vec, result):
         print(f"向量数据：{vec} 预测维度：{torch.argmax(res).item()}, 真实维度：{np.argmax(vec)}, 预测概率：{torch.max(res).item():6f}, 预测结果：{"正确" if np.argmax(vec) == torch.argmax(res) else "错误" }")  # 打印结果
         count += 1
         if np.argmax(vec) != torch.argmax(res):
            error += 1
    accuracy = (count - error) / count
    print(f"本次模型预测成功率:{accuracy:.2%}")

if __name__ == "__main__":
    main()
    # 生成测试数据
    X,Y = build_dataset(20)
    predict("model.bin", X)
