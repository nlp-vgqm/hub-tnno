import os
# 1. 先设置环境变量解决OMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果哪一维大就输出哪一维

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.softmax # nn.softmax() softmax将多分类的结果以概率的形式展示，且概率和相加为1，最终选取概率值最大的分类 作为最终结果
        self.loss = nn.CrossEntropyLoss()  # loss函数采用多分类交叉熵损失函数，其中包含了softmax激活函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)
        y_pred = x
        if y is not None:
            # y需要是长整型，形状为(batch_size,)
            return self.loss(y_pred, y.long())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，返回最大的值的索引
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # ✅ 修复：先转成numpy数组，再转张量
    X_np = np.array(X)  # 形状: (total_sample_num, 5)
    Y_np = np.array(Y)  # 形状: (total_sample_num, 1)

    return torch.FloatTensor(X_np), torch.FloatTensor(Y_np)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() #评估模式而不是训练模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print(x.shape, y.shape)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测  返回概率分布
        # 取概率最大的索引作为预测类别
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y).sum().item()

    accuracy = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


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
            watch_loss.append(loss.item()) # loss是一个张量，loss.item()是转换为数字常量
        # print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # print(watch_loss)
        acc = evaluate(model)  # 测试本轮模型结果
        print(acc, float(np.mean(watch_loss)))
        log.append([acc, float(np.mean(watch_loss))])
    #     # for name, param in model.named_parameters():
    #     #     print(f"{name}:")
    #     #     print(f"  形状: {param.shape}")
    #     #     print(f"  值: {param.data}")
    #     #     print(f"  是否可训练: {param.requires_grad}")
    # 保存模型
    torch.save(model.state_dict(), "model2.bin")
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

        logits = model(torch.FloatTensor(input_vec)) #返回概率分布
        probabilities = torch.softmax(logits, dim=1)  # 改成用softmax得到概率
        predicted = torch.argmax(probabilities, dim=1)

    for i, (vec, pred, prob_dist) in enumerate(zip(input_vec, predicted, probabilities)):
        confidence = prob_dist[pred].item()
        print(f"输入{i+1}: {vec}")
        print(f"  预测索引: {pred.item()}, 置信度: {confidence:.4f}")
        print(f"  概率分布: [{', '.join([f'{p:.2f}' for p in prob_dist.tolist()])}]")



if __name__ == "__main__":
    # main()
    test_vec = [[0.97889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model2.bin", test_vec)
    print(build_sample())
