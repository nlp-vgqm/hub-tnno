# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        # 使用平均池化
        x = x.mean(dim=1)
        x = self.layer(x)
        x = self.dropout(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.margin = config.get("margin", 0.2)  # 三元组损失margin参数
        # 保留原有的CosineEmbeddingLoss用于二分类任务
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def encode(self, sentence):
        """编码句子"""
        return self.sentence_encoder(sentence)

    def cosine_similarity(self, tensor1, tensor2):
        """计算余弦相似度"""
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(tensor1 * tensor2, dim=-1)
        return cosine

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离 = 1 - 余弦相似度"""
        return 1 - self.cosine_similarity(tensor1, tensor2)

    def triplet_loss(self, anchor, positive, negative, margin=None):
        """
        三元组损失
        anchor: 锚点句子表示
        positive: 正例句子表示
        negative: 负例句子表示
        margin: 边界值
        """
        if margin is None:
            margin = self.margin

        # 计算正负样本距离
        pos_dist = self.cosine_distance(anchor, positive)
        neg_dist = self.cosine_distance(anchor, negative)

        # 三元组损失: max(0, pos_dist - neg_dist + margin)
        loss = torch.relu(pos_dist - neg_dist + margin)

        # 返回平均损失
        return loss.mean()

    def forward(self, sentence1, sentence2=None, sentence3=None, target=None, mode="triplet"):
        """
        前向传播
        支持多种模式:
        - triplet: 三元组训练 (需要sentence1, sentence2, sentence3)
        - pair: 句子对训练 (需要sentence1, sentence2, target)
        - encode: 仅编码句子 (只需要sentence1)
        """
        if mode == "triplet":
            # 三元组训练模式
            assert sentence2 is not None and sentence3 is not None, \
                "三元组训练需要三个句子输入"

            anchor = self.encode(sentence1)
            positive = self.encode(sentence2)
            negative = self.encode(sentence3)

            return self.triplet_loss(anchor, positive, negative)

        elif mode == "pair":
            # 句子对训练模式 (二分类)
            assert sentence2 is not None, "句子对训练需要两个句子输入"

            vector1 = self.encode(sentence1)
            vector2 = self.encode(sentence2)

            if target is not None:
                return self.cosine_loss(vector1, vector2, target.squeeze())
            else:
                # 计算余弦相似度用于推理
                return self.cosine_similarity(vector1, vector2)

        elif mode == "encode":
            # 仅编码句子
            return self.encode(sentence1)

        else:
            raise ValueError(f"不支持的mode: {mode}")


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)