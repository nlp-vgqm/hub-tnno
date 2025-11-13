# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json


"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中按照汉字出现的维度进行多分类任务

"""


class TorchModel(nn.Module):
    def __init__(self,vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size=128, bias=False, batch_first=True)  # 线性层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.classify = nn.Linear(128,7)
        self.loss = nn.functional.cross_entropy # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        x = output.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        if y is not None:
            return self.loss(x,y.long())  # 预测值和真实值计算损失
        else:
            return x # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他ABCD"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定字符出现返回索引位置
    tag = "你我他"
    y = -1
    for i, char in enumerate(x):
        if char in tag:
            y = i
            break
        else:
            y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
        # print(x)
        # print(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted = torch.argmax(y_pred, dim=1)  # 获取预测类别
        correct = (predicted == y.squeeze()).sum().item()
        wrong = len(y) - correct
    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.2%}" )
    return accuracy

def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 构建词表
    vocab = build_vocab()
    print(vocab)
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x,y)  # 计算loss
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
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    # model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 修复安全警告
    x = []
    for input_string in input_strings:
        # 使用get方法避免KeyError，对于不在词表中的字符使用'unk'
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        predicted = torch.argmax(result[i]).item()  # 找到概率最大的类别
        probabilities = torch.softmax(result[i], dim=0)[predicted].item()  # 获取对应的概率值
        print(f"输入：{input_string}, 预测类别：{predicted}, 概率值：{probabilities}" )  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = ["fnf我ye", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
