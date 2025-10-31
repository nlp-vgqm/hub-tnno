# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""

基于pytorch框架编写模型训练
实现一个多分类任务：五维随机向量最大的数字在哪维就属于哪一类
共5个类别（0-4维）

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 128)  # 增加隐藏层
        self.activation = torch.sigmoid
        self.output_layer = nn.Linear(128, num_classes)  # 输出5个类别的分数, 添加此层后准确率提升较快
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 128)
        x = self.activation(x)
        x = self.output_layer(x)  # (batch_size, 128) -> (batch_size, 5)
        
        if y is not None:
            return self.loss(x, y.squeeze().long())  # 交叉熵损失需要long类型的标签
        else:
            return torch.softmax(x, dim=1)  # 输出概率分布

# 生成一个样本：五维随机向量，标签为最大值所在的维度
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找到最大值的索引
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别数量
    class_counts = [0] * 5
    for label in y:
        class_counts[int(label)] += 1
    print("各类别样本数量:", class_counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 获取概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测类别
        true_classes = y.squeeze().long()
        
        for pred, true in zip(predicted_classes, true_classes):
            if pred == true:
                correct += 1
            else:
                wrong += 1
                
    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 错误预测个数：{wrong}, 正确率：{accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 50  # 增加训练轮数
    batch_size = 32
    train_sample = 10000  # 增加训练样本数量
    input_size = 5
    num_classes = 5  # 5个类别
    learning_rate = 0.001
    
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
        
        # 打乱数据
        indices = torch.randperm(train_sample)
        train_x_shuffled = train_x[indices]
        train_y_shuffled = train_y[indices]
        
        for batch_index in range(train_sample // batch_size):
            start_idx = batch_index * batch_size
            end_idx = (batch_index + 1) * batch_size
            x = train_x_shuffled[start_idx:end_idx]
            y = train_y_shuffled[start_idx:end_idx]
            
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        
    
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(avg_loss)])
    
    # 保存模型
    torch.save(model.state_dict(), "model_classification.bin")
    
    # 画图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], 'b-', label="acc", linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('epoch-accuracy diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], 'r-', label="loss", linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('epoch-loss diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    with torch.no_grad():
        # 获取概率分布和预测类别
        probabilities = model.forward(torch.FloatTensor(input_vec))
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    for i, (vec, probs, pred_class) in enumerate(zip(input_vec, probabilities, predicted_classes)):
        max_index_true = np.argmax(vec)  # 真实的最大值位置
        confidence = probs[pred_class].item()  # 预测类别的置信度
        is_correct = "✓" if pred_class == max_index_true else "✗"
        
        print(f"样本 {i+1}:")
        print(f"  输入向量: {[f'{x:.4f}' for x in vec]}")
        print(f"  真实最大值位置: 第{max_index_true}维")
        print(f"  预测最大值位置: 第{pred_class}维 {is_correct}")
        print(f"  预测置信度: {confidence:.4f}")
        print(f"  各类别概率: {[f'{p:.4f}' for p in probs]}")

if __name__ == "__main__":
    main()
    
    # 测试样例
    print("\n" + "="*50)
    print("模型预测测试:")
    print("="*50)
    
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.1],  # 最大值在第0维
        [0.1, 0.8, 0.7, 0.6, 0.9],  # 最大值在第4维
        [0.2, 0.3, 0.95, 0.4, 0.5], # 最大值在第2维
        [0.1, 0.9, 0.8, 0.7, 0.6],  # 最大值在第1维
        [0.3, 0.4, 0.5, 0.6, 0.2],  # 最大值在第3维
        [0.15, 0.25, 0.35, 0.45, 0.55],  # 边界情况：接近的值
        [0.95, 0.94, 0.93, 0.92, 0.91],  # 边界情况：很接近的值
    ]
    predict("model_classification.bin", test_vec)