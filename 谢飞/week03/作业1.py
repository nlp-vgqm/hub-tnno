#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的RNN网络编写
实现一个网络完成NLP任务
判断文本中特定字符的位置

"""

class RNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)  # RNN层，双向
        self.classify = nn.Linear(hidden_size * 2, sentence_length + 1)  # 线性层，输出每个位置的概率 + 无特定字符的情况
        self.activation = nn.Softmax(dim=-1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size*2)
        
        # 取最后一个时间步的输出，或者可以使用所有时间步输出的均值/最大值
        # 这里我们使用最后一个时间步的输出
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size*2)
        
        x = self.classify(last_output)  # (batch_size, hidden_size*2) -> (batch_size, sentence_length+1)
        y_pred = self.activation(x)  # (batch_size, sentence_length+1)
        
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

# 字符集
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    # 查找特定字符（"你","我","他"）的位置
    target_chars = {"你", "我", "他"}
    positions = [i for i, char in enumerate(x) if char in target_chars]
    
    if positions:
        # 如果有特定字符，标签为第一个特定字符的位置（0-indexed）
        y = min(positions)  # 取第一个出现的位置
    else:
        # 如果没有特定字符，标签为sentence_length（表示未出现）
        y = sentence_length
    
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = RNNModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    
    # 统计各类别的数量
    class_counts = {i: 0 for i in range(sentence_length + 1)}
    for label in y:
        class_counts[label.item()] += 1
    
    print("各类别样本数量：", class_counts)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        
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
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 50  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    learning_rate = 0.001  # 学习率
    
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        
        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        
        print("=========\n第%d轮平均loss:%f, 准确率:%f" % (epoch + 1, avg_loss, acc))

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 50  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    
    x = []
    for input_string in input_strings:
        # 将输入序列化，如果长度不够则填充，如果太长则截断
        encoded = [vocab.get(char, vocab['unk']) for char in input_string]
        if len(encoded) < sentence_length:
            encoded += [0] * (sentence_length - len(encoded))  # 填充
        else:
            encoded = encoded[:sentence_length]  # 截断
        x.append(encoded)
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    
    for i, input_string in enumerate(input_strings):
        probs = result[i]
        predicted_class = torch.argmax(probs).item()
        
        if predicted_class == sentence_length:
            position_info = "未出现特定字符"
        else:
            position_info = f"第{predicted_class + 1}个位置"
        
        # 显示每个位置的置信度
        confidence_info = {f"pos_{j+1}": f"{probs[j].item():.3f}" for j in range(sentence_length)}
        confidence_info["none"] = f"{probs[sentence_length].item():.3f}"
        
        print(f"输入：{input_string}")
        print(f"预测结果：{position_info}")
        print(f"位置置信度：{confidence_info}")
        print("-" * 50)

if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww", "你我他abc", "xyzabcde"]
    predict("rnn_model.pth", "vocab.json", test_strings)
