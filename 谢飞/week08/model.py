# -*- coding: utf-8 -*-

"""
模型定义
包含句子编码器、三元组损失函数和三元组网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    """句子编码器：将输入文本编码为固定维度的向量"""
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # 使用LSTM或线性层
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        输入: x (batch_size, max_length)
        输出: x (batch_size, hidden_size)
        """
        x = self.embedding(x)  # (batch_size, max_length, hidden_size)
        # 使用LSTM
        x, _ = self.lstm(x)  # (batch_size, max_length, hidden_size * 2)
        # 使用最大池化
        x = F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()  # (batch_size, hidden_size * 2)
        x = self.layer(x)  # (batch_size, hidden_size)
        x = self.dropout(x)
        return x


class TripletLoss(nn.Module):
    """三元组损失函数
    
    三元组损失的目标是：让anchor和positive之间的距离小于anchor和negative之间的距离，
    且差距至少为margin。即：d(anchor, positive) + margin < d(anchor, negative)
    
    损失函数：L = max(0, d(a,p) - d(a,n) + margin)
    """
    def __init__(self, margin=0.5, distance_type='euclidean'):
        """
        Args:
            margin: 正负样本之间的最小距离间隔
            distance_type: 距离类型，'euclidean'（欧氏距离）或'cosine'（余弦距离）
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_type = distance_type

    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        Args:
            anchor: 锚点向量 (batch_size, hidden_size)
            positive: 正样本向量 (batch_size, hidden_size)
            negative: 负样本向量 (batch_size, hidden_size)
        Returns:
            loss: 三元组损失值
        """
        if self.distance_type == 'euclidean':
            # 计算欧氏距离
            distance_positive = F.pairwise_distance(anchor, positive, p=2)
            distance_negative = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_type == 'cosine':
            # 计算余弦距离（1 - 余弦相似度）
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positive_norm = F.normalize(positive, p=2, dim=1)
            negative_norm = F.normalize(negative, p=2, dim=1)
            
            # 余弦相似度
            cos_sim_pos = (anchor_norm * positive_norm).sum(dim=1)
            cos_sim_neg = (anchor_norm * negative_norm).sum(dim=1)
            
            # 余弦距离 = 1 - 余弦相似度
            distance_positive = 1 - cos_sim_pos
            distance_negative = 1 - cos_sim_neg
        else:
            raise ValueError("distance_type must be 'euclidean' or 'cosine'")
        
        # 三元组损失：max(0, d(a,p) - d(a,n) + margin)
        # 我们希望 d(a,p) < d(a,n)，即 d(a,n) - d(a,p) > margin
        # 如果 d(a,p) - d(a,n) + margin < 0，说明已经满足条件，损失为0
        loss = F.relu(distance_positive - distance_negative + self.margin)
        
        # 返回平均损失
        return loss.mean()


class TripletNetwork(nn.Module):
    """基于三元组损失的文本匹配网络"""
    def __init__(self, config):
        super(TripletNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.triplet_loss = TripletLoss(
            margin=config.get("margin", 0.5),
            distance_type=config.get("distance_type", "euclidean")
        )

    def forward(self, anchor, positive=None, negative=None):
        """
        前向传播
        Args:
            anchor: 锚点文本 (batch_size, max_length)
            positive: 正样本文本 (batch_size, max_length)，可选
            negative: 负样本文本 (batch_size, max_length)，可选
        Returns:
            如果提供了positive和negative，返回损失值
            否则只返回anchor的编码向量
        """
        anchor_vec = self.sentence_encoder(anchor)
        
        if positive is not None and negative is not None:
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            loss = self.triplet_loss(anchor_vec, positive_vec, negative_vec)
            return loss
        else:
            return anchor_vec

    def encode_sentence(self, sentence):
        """编码单个句子"""
        self.eval()
        with torch.no_grad():
            return self.sentence_encoder(sentence)

