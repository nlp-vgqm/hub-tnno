# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类
"""

class TorchModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            result = []
            num = np.array(y_pred).size / len(y_pred)   # 预测值的维度
            for y_p in zip(y_pred):
                one_hot = np.zeros(int(num))            # 生成一个与预测值同维度的全为0的向量
                max_index = np.argmax(y_p)              # 预测值中最大数据对应的位数
                one_hot[max_index] = 1                  # 将全为0的向量中与预测值中最大数据对应的位数相同的位置值修改为1
                result.append(one_hot)
            return y_pred,result

# 生成样本数据
# 随机生成一个5维向量，如果哪一位最大就表示第几类
def build_sample():
    x = np.random.random(5)    # 随机生成5维样本数据
    max_index = np.argmax(x)   # 找出数据最大的位数
    one_hot = np.zeros(5)      # 生成5维全是数字0的向量
    one_hot[max_index] = 1     # 给全为0的向量中与样本数据中数字最大的位数相同的位置赋值为1
    return x,one_hot

# 批量生成训练样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码，测试每轮训练，模型的正确率
def evaluate(model):
    model.eval()    # 测试模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print(x,y)
    correct, worng = 0, 0
    with torch.no_grad():
        y_pred,result = model(x)
        for y_p, y_t in zip(y_pred,y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1    # 预测正确
            else:
                worng += 1      # 预测错误
    print("预测正确的个数%d,预测错误的个数%d,正确率为%f" % (correct,worng,correct / (correct + worng)))
    return correct / (correct + worng)

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size,output_size)
    model.load_state_dict(torch.load(model_path))   # 加载训练好的权重
    #print(model.state_dict())

    model.eval()    #测试模式
    with torch.no_grad():
        y_pred, result = model.forward(torch.FloatTensor(input_vec))
    for vec, y_p, res in zip(input_vec, y_pred, result):
        print("输入：%s, 预测值:%s, 预测分类为:%s" % (vec, y_p, res))

def main():
    # 配置参数
    epoch_num = 30             # 定义训练轮次
    batch_size = 20            # 定义每次训练的样本个数
    train_sample = 5000        # 定义每轮训练总共训练的样本总数
    input_size = 5             # 输入向量维数
    output_size = 5            # 输出向量维数
    lerning_rate = 0.01        # 设置学习率

    # 简历模型
    model = TorchModel(input_size,output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=lerning_rate)
    #
    log = []
    # 创建训练集
    tarin_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = tarin_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x,y)   # 计算loss
            loss.backward()     # 计算梯度
            optim.step()        # 权重更新
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
        print("================")
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(),"model.bin")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label='acc')     # 画正确率曲线
    plt.plot(range(len(log)), [i[1] for i in log], label='loss')    # 画损失值曲线
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
