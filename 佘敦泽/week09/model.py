# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF # pip install pytorch-crf
from transformers import BertModel

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, hidden_size * 2)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict

# 如果使用bert，则使用BertTorchModel
class BertTorchModel(nn.Module):
    def __init__(self, bert_model:BertModel, config):
        super(BertTorchModel, self).__init__()
        self.bert_model = bert_model
        for param in self.bert_model.parameters(): # 这里冻结 bert 预训练模型的参数
            param.requires_grad_ = False

        self.hidden_size = bert_model.config.hidden_size
        self.vocab_size = bert_model.config.vocab_size # +1 ?
        self.embedding = bert_model.embeddings.word_embeddings

        class_num = config["class_num"]

        # NOTE: 如果使用了 bert 模型来提取特征, 可以将 bilstm 模型来提取特征的方式注释掉
        # num_layers = config["num_layers"]
        # self.layer = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)

        self.classify = nn.Linear(self.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # 1. 直接经过 bert 模型 -- NOTE: 注意需要和 __init__ 中的参数对应
        bert_output = self.bert_model(x) # 让输入经过bert, 再经过 bilstm 层做后续逻辑处理
        x = bert_output.last_hidden_state

        # 2. 只经过 Bert embedding 层
        # x = self.embedding(x)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, hidden_size * 2)

        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
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