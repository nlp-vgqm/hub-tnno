#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
使用rnn模型，判断特定字符在文本中的位置。多标签分类，特定字符所在位置标记为1，未出现的标记为0

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnnlayer = nn.RNN(sentence_length, hidden_size, batch_first=True)
        self.fclayer = nn.Linear(hidden_size, 16)     #线性层
        self.fclayer2 = nn.Linear(16, sentence_length)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵损失
        self.loss = nn.BCEWithLogitsLoss()  # loss函数采用bce损失
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # print('enbedding后', x.shape)
        output, _ = self.rnnlayer(x)                 # output:(batch_size, sen_len, hidden_size)
        x = output[:, -1, :]                         # x: (batch_size, hidden_size)
        x = self.fclayer(x)                       #(batch_size, hidden_size) -> (batch_size, sentence_length) 3*20 20*1 -> 3*1
        y_pred = self.fclayer2(x)
        # print('fc后', y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            #return torch.softmax(y_pred, dim=-1)                 #输出预测结果
            return torch.sigmoid(y_pred)


#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    """
    生成文本样本，返回每个位置是否包含目标字符
    Args:
        vocab: 词汇表字典，格式为 {字符: 索引}
        sentence_length: 生成的句子长度
    Returns:
        x: 字符索引序列，形状为 [sentence_length]
        y: 位置标签序列，形状为 [sentence_length]，1表示目标字符位置，0表示其他
    """
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    target_chars = ['你', '我', '他']
    y = []
    for char in x:
        # 如果是目标字符，标记为1，否则为0
        if char in target_chars:
            y.append(1)
        else:
            y.append(0)

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

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length, sample_length=200):
    model.eval()
    x, y = build_dataset(sample_length, vocab, sentence_length)   #建立200个用于测试的样本

    with torch.no_grad():
        y_pred = model(x)      #模型预测
        y_pred= y_pred.tolist()
        y = y.tolist()

        collect_count = 0
        for row_p, row_t in zip(y_pred,y):
            y_p_new = []

            for value_p in row_p:
                if float(value_p) >= 0.9:
                    y_p_new.append(1)
                else:
                    y_p_new.append(0)
            if y_p_new == row_t:
                collect_count += 1

    print("正确预测个数：%d, 正确率：%f" % (collect_count, collect_count / sample_length))
    return collect_count / sample_length


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 5000    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.001 #学习率
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
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model1.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化

    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        y_p = model.forward(torch.LongTensor(x))  #模型预测

    y_p = y_p.tolist()

    for input, p_v in zip(input_strings, y_p):
        result_p_index = []
        result_p_element = []
        for p_v_v in p_v:
            if float(p_v_v) >= 0.9:
                result_p_index.append(p_v.index(p_v_v))
                result_p_element.append(p_v_v)

        print("输入：%s,  特定字符'你我他'所在位置索引：%s, 概率值：%s" % (input, result_p_index, result_p_element))



if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你d我g", "aqwdeg", "n我kwww"]
    predict("model1.pth", "vocab.json", test_strings)
