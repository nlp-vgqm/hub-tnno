#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
对文本里的字符进行分类，字符出现在文本里的哪一个下标，就代表为哪一类，若没有在文本里出现，则代表为len+1类

"""

class WordClassifyModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(WordClassifyModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.layer = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True)   #用RNN来作语义池化
        self.classify = nn.Linear(vector_dim, len(vocab)+1)     #线性层
        self.loss = nn.CrossEntropyLoss()  #loss函数采用交差熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, x = self.layer(x)                   # output:(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim) x:(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        x = x.squeeze()                             # 把维度为1的去掉
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, sen_len)
        if y is not None:
            return self.loss(x, y)   #预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=-1)                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "我们都是中华人民共和国建设者"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) 
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#计算这个随机样本在字典字符的下标
def build_sample(vocab, sentence_length):
    keys = list(vocab.keys())
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(keys) for _ in range(sentence_length)]
    #判断x在vocab出现的下标
    y = find_str_index(keys, x)
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, build_one_list(len(vocab)+1, y)

# 获取文本x在vocab字典里的下标值
def find_str_index(keys, x):
    keys_str = ''.join([str(item) for item in keys])
    keys_str.replace('pad', 'p')
    x_str = ''.join([str(item) for item in x])
    x_str.replace('pad', 'p')
    index = keys_str.find(x_str)
    if index != -1:
        return index
    else:
        return len(keys)

# 生成连续的文本，从vocab里取连续的3个字符作为训练样本
def build_substr(vocab, sentence_length):
    keys = list(vocab.keys())
    startIndex = random.randint(0, len(keys)-sentence_length)
    x = keys[startIndex:startIndex + sentence_length]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x,build_one_list(len(vocab)+1, startIndex)

# 生成全零矩阵，并设置index下标对应值为1
def build_one_list(lenth, index):
    A = np.zeros(lenth)
    A[index] = 1
    return A

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        # 随机样本，基本上很难成为字典字符的完整子集
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
        # 连续子样本，是字典字符的完整子集
        x, y = build_substr(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = WordClassifyModel(char_dim, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            # print(y_p,np.argmax(y_p),y_t,y_t[np.argmax(y_p)],y_t[np.argmax(y_p)],'数据对比')
            if y_t[np.argmax(y_p)] == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 3   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim)
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
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def find_str_index2(vocab, input_str):
    keys = list(vocab.keys())
    keys_str = ''.join([str(item) for item in keys])
    keys_str = keys_str.replace('pad', 'p')
    return keys_str.find(input_str.replace('pad', 'p'))

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  #将输入序列化
    tensor_x = torch.LongTensor(x)
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(tensor_x)  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 在文本中下标为：%d, 预测结果为：%f 类" % (input_string, find_str_index2(vocab, input_string), int(np.argmax(result[i])))) #打印结果

if __name__ == "__main__":
    # main()
    # "我们都是中华人民共和国建设者"
    test_strings = ["我们都", "都是中", "我是中", "abd", "建设者", "我中国", "是中华", "共和国", "天天好", "人民共", "xyz"]
    predict("model.pth", "vocab.json", test_strings)
