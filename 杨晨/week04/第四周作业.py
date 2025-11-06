# coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0)  # shape=(vocab_size, dim)
        self.rnn_layer = nn.RNN(input_size=input_dim,
                                hidden_size=hidden_size,
                                batch_first=True,
                                num_layers=num_rnn_layers,
                                )
        self.classify = nn.Linear(hidden_size, 2)  # w = hidden_size * 2
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  # output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)  # output shape:(batch_size, sen_len, 2) -> y_pred.view(-1, 2) (batch_size*sen_len, 2)
        if y is not None:
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))
        else:
            return y_pred


class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()

    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                # 使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    # 将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


# 文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence


# 基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label


# 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab


# 建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length)  # diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)  # torch
    return data_loader


# ============ 全切分算法实现 ============
def all_cut(sentence, Dict):
    """
    全切分函数 - 基于回溯算法找出句子所有可能的切分方式
    要求每个切分出来的词都在词典中

    参数:
        sentence: 待切分的中文字符串
        Dict: 词典，键为词，值为词频

    返回:
        包含所有可能切分方式的列表，每个切分方式是一个词列表
    """
    results = []  # 存储所有切分结果
    current = []  # 存储当前切分路径

    def backtrack(start_index):
        """
        回溯函数，递归查找所有可能切分
        start_index: 当前处理到的字符位置
        """
        # 如果已经处理到句子末尾，将当前切分结果加入最终结果
        if start_index == len(sentence):
            results.append(current[:])  # 使用切片复制当前列表
            return

        # 尝试所有可能的切分长度
        for end_index in range(start_index + 1, len(sentence) + 1):
            word = sentence[start_index:end_index]

            # 如果当前词在词典中，继续递归切分
            if word in Dict:
                current.append(word)  # 选择当前词
                backtrack(end_index)  # 递归处理剩余部分
                current.pop()  # 撤销选择，回溯

    backtrack(0)  # 从第一个字符开始回溯
    return results


def test_all_cut():
    """
    测试全切分函数
    """
    # 测试词典和句子
    Dict = {"经常": 0.1,
            "经": 0.05,
            "有": 0.1,
            "常": 0.001,
            "有意见": 0.1,
            "歧": 0.001,
            "意见": 0.2,
            "分歧": 0.2,
            "见": 0.05,
            "意": 0.05,
            "见分歧": 0.05,
            "分": 0.1}

    sentence = "经常有意见分歧"

    # 获取所有切分结果
    all_segmentations = all_cut(sentence, Dict)

    # 输出结果
    print(f"句子: '{sentence}'")
    print(f"共有 {len(all_segmentations)} 种切分方式:")
    for i, seg in enumerate(all_segmentations, 1):
        print(f"{i:2d}. {'/'.join(seg)}")

    # 验证结果是否与预期一致
    expected_results = [
        ['经常', '有意见', '分歧'],
        ['经常', '有意见', '分', '歧'],
        ['经常', '有', '意见', '分歧'],
        ['经常', '有', '意见', '分', '歧'],
        ['经常', '有', '意', '见分歧'],
        ['经常', '有', '意', '见', '分歧'],
        ['经常', '有', '意', '见', '分', '歧'],
        ['经', '常', '有意见', '分歧'],
        ['经', '常', '有意见', '分', '歧'],
        ['经', '常', '有', '意见', '分歧'],
        ['经', '常', '有', '意见', '分', '歧'],
        ['经', '常', '有', '意', '见分歧'],
        ['经', '常', '有', '意', '见', '分歧'],
        ['经', '常', '有', '意', '见', '分', '歧']
    ]

    # 检查结果数量
    print(f"\n预期切分方式数量: {len(expected_results)}")
    print(f"实际切分方式数量: {len(all_segmentations)}")
    print(f"结果验证: {'通过' if len(all_segmentations) == len(expected_results) else '不通过'}")

    return all_segmentations


def main():
    """
    主函数 - 可以选择运行神经网络训练或全切分测试
    """
    print("选择运行模式:")
    print("1. 神经网络分词模型训练")
    print("2. 全切分算法测试")

    choice = input("请输入选择 (1 或 2): ")

    if choice == "1":
        # 神经网络训练模式
        epoch_num = 5  # 训练轮数
        batch_size = 20  # 每次训练样本个数
        char_dim = 50  # 每个字的维度
        hidden_size = 100  # 隐含层维度
        num_rnn_layers = 1  # rnn层数
        max_length = 20  # 样本最大长度
        learning_rate = 1e-3  # 学习率
        vocab_path = "chars.txt"  # 字表文件路径
        corpus_path = "../corpus.txt"  # 语料文件路径
        vocab = build_vocab(vocab_path)  # 建立字表
        data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  # 建立数据集
        model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)  # 建立模型
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
        # 训练开始
        for epoch in range(epoch_num):
            model.train()
            watch_loss = []
            for x, y in data_loader:
                optim.zero_grad()  # 梯度归零
                loss = model.forward(x, y)  # 计算loss
                loss.backward()  # 计算梯度
                optim.step()  # 更新权重
                watch_loss.append(loss.item())
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 保存模型
        torch.save(model.state_dict(), "model.pth")

    elif choice == "2":
        # 全切分算法测试模式
        test_all_cut()
    else:
        print("无效选择，运行全切分算法测试")
        test_all_cut()


# 最终预测
def predict(model_path, vocab_path, input_strings):
    # 配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 1  # rnn层数
    vocab = build_vocab(vocab_path)  # 建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        # 逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  # 预测出的01序列
            # 在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()


if __name__ == "__main__":
    # 默认运行全切分测试
    test_all_cut()

    # 如果需要运行神经网络预测，取消下面的注释
    # input_strings = ["同时，国内有望出台，新汽车刺激方案",
    #                  "沪胶后市有望延续强势！",
    #                  "经过两个交易日的强势调整后",
    #                  "昨日上海天然橡胶期货价格再度大幅上扬"]
    # predict("model.pth", "chars.txt", input_strings)
