#coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import random

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, 7)     #线性层
        # self.activation = torch.sigmoid     #sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  #loss函数采用均方差损失
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, hidden = self.rnn(x)
        x = output[:, -1, :]
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*20 20*1 -> 3*1
        # y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(x, y.squeeze().long())   #预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)                 #输出预测结果

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    if set("我") & set(x):
            y = [i for i, char in enumerate(x) if set(char) == set("我")]
            y = y[0]
    else:
        y = 6
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(20000, vocab, sample_length)  # 建立20000个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), 20000 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 得到概率分布，形状: (100, 5)
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        true_classes = y.squeeze().long()  # 确保标签形状匹配
        print(true_classes)

        correct = (predicted_classes == true_classes).sum().item()
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_string in input_strings:
        if len(input_string) < sentence_length:
            padded_string = input_string + "pad" * (sentence_length - len(input_string))
        else:
            padded_string = input_string[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in padded_string])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 形状: (batch_size, sentence_length + 1)
        predicted_classes = torch.argmax(result, dim=1)  # 预测的类别索引

    for input_str, pred_class, probs in zip(input_strings, predicted_classes, result):
        if pred_class.item() == sentence_length:
            position_info = "不包含'我'"
        else:
            position_info = f"第{pred_class.item() + 1}个位置是'我'"

        # 获取最高概率值
        max_prob = torch.max(probs).item()
        print("输入：%s, 预测结果：%s, 最高概率：%.4f" % (input_str, position_info, max_prob))


if __name__ == "__main__":
    main()
    test_strings = ["我n我fee", "wz你dfg", "rqwd我g", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
