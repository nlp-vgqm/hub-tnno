# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        class_num = config["class_num"]
        # num_layers = config["num_layers"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(768, class_num)

        
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x, _ = self.encoder(x)  # x shape: (batch_size, seq_len, 768)
        # 跳过CLS token，从第1个位置开始进行序列标注
        predict = self.classify(x)  # shape: (batch_size, seq_len-1, class_num)
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
        # 分层学习率：BERT层用小学习率，分类层用相对较大的学习率
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': learning_rate},
            {'params': model.classify.parameters(), 'lr': learning_rate * 10}
        ]
        return Adam(param_groups)
    elif optimizer == "sgd":
        param_groups = [
            {'params': model.encoder.parameters(), 'lr': learning_rate},
            {'params': model.classify.parameters(), 'lr': learning_rate * 10}
        ]
        return SGD(param_groups)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)