# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=config.get("triplet_margin", 0.1),
                                                 p=2, reduction='mean')

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        # 计算a和p的余弦距离
        ap = self.cosine_distance(a, p)
        # 计算a和n的余弦距离
        an = self.cosine_distance(a, n)
        # 如果没有设置margin，则设置diff为ap - an + 0.1
        if margin is None:
            diff = ap - an + 0.1
        # 如果设置了margin，则设置diff为ap - an + margin.squeeze()
        else:
            diff = ap - an + margin.squeeze()
        # 返回diff中大于0的部分的平均值
        return torch.mean(diff[diff.gt(0)])  # greater than

    # sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None, target=None, mode='pair'):
        """
        参数说明:
        - mode: 'pair'表示二元组训练，'triplet'表示三元组训练
        - sentence1: anchor (batch_size, max_length)
        - sentence2: positive (batch_size, max_length) 或 正样本
        - sentence3: negative (batch_size, max_length) 或 负样本
        - target: 仅用于二元组训练
        """
        if mode == 'triplet':
            # 三元组训练模式
            anchor = self.sentence_encoder(sentence1)
            positive = self.sentence_encoder(sentence2)
            negative = self.sentence_encoder(sentence3)
            return self.triplet_loss(anchor, positive, negative)

        # 二元组训练模式（保持原有逻辑）
        elif mode == 'pair':
            # 同时传入两个句子
            if sentence2 is not None:
                vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
                vector2 = self.sentence_encoder(sentence2)
                # 如果有标签，则计算loss
                if target is not None:
                    return self.loss(vector1, vector2, target.squeeze())
                # 如果无标签，计算余弦距离
                else:
                    return self.cosine_distance(vector1, vector2)
            # 单独传入一个句子时，认为正在使用向量化能力
            else:
                return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["triplet_margin"] = 0.1
    model = SiameseNetwork(Config)

    # 测试三元组损失
    anchor = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    positive = torch.LongTensor([[1, 2, 3, 4], [2, 2, 3, 4]])
    negative = torch.LongTensor([[3, 2, 1, 0], [4, 4, 0, 0]])
    loss = model(anchor, positive, negative, mode='triplet')
    print(f"Triplet Loss: {loss}")

    # 测试原有功能
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [2, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, target=l, mode='pair')
    print(f"Pair Loss: {y}")
