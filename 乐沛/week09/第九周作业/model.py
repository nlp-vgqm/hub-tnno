# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained("bert-base-chinese", output_hidden_states=True)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # 原代码：x = self.embedding(x)
        # 原代码：x, _ = self.layer(x)

        # 使用BERT替代原来的embedding+LSTM
        # 注意：BERT需要attention_mask，假设输入x已经包含了padding信息
        # 创建attention_mask（假设0是padding）
        attention_mask = (x != 0).long()

        # BERT前向传播
        bert_outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 使用BERT的最后一层隐藏状态
        x = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 可选：应用dropout
        x = self.dropout(x)

        predict = self.classify(x)  # (batch_size, sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                # 使用attention_mask作为CRF的mask
                mask = attention_mask.bool()
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # 创建mask用于解码
                mask = (x != 0).any(dim=-1)  # 或者使用attention_mask
                return self.crf_layer.decode(predict, mask=mask)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
