# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    # 自定义embedding层的向量维度和句子的长度。传参，其中前三个为embedding函数的参数，后面1个是RNN的参数，inputsize的参数即为embedding层的输出。
    #vector_dim是转向量的维度，sentence_length是句子长度，vocab是句子。
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.loss = nn.functional.mse_loss

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x=self.embedding(x)   # embedding层。输出为batch*sentence_length*vector_dim的张量
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()       #池化后，输出为batch*1*vector_dim的张量
        print(x.shape)
        output,h = self.layer(x)
        y_pred=output.detach()
        print(f'{y_pred=}')
        #y_pred=torch.FloatTensor(y_pred)
        #print(f'{y_pred=}')
        y_pred = torch.softmax(y_pred, dim=0)
        print(f'{y_pred=}')
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=0)  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他de"  # 字符集
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
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]    #生成的是一个字符列表，表示一个文本
    # 如果“我”这个字符出现在哪，则为哪一类，否则返回0
    y=[0]*sentence_length   #如果都没有这个字符出现，则返回全0
    for i, char_ in enumerate(x):
        if "我" == char_:
            y[i]=1     #如果“我”这个字符在哪，这个下标的值变为1,找到第一个出现“我”的地方
            break
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
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    print("本次预测集中共有200个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.all(y_t == 0):
                pass
            else:
                if torch.argmax(y_p) == torch.argmax(y_t):
                    correct += 1
                else:
                    wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 5  # 每次训练样本个数
    train_sample = 50  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.00001  # 学习率
    # 建立字表
    vocab = build_vocab()    #vocab是一个字典，字符串集
    print(f'{vocab=}')
    hidden_size=sentence_length
    #print(f'{vocab=}')
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length,hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            print(f'{x=},{y=}')
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 训练完一轮后，查看训练的结果，是重新用新的样本来测试本轮模型结果
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
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i]))  # 打印结果


if __name__ == "__main__":
    main()

