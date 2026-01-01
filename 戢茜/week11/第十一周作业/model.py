# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertTokenizer, BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config,pretrain_model_path):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, class_num)
        self.pool = nn.AvgPool1d(max_length)
        self.activation = torch.relu     #relu做激活函数
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if target is not None:
            #mask_y = np.ones(y.shape[0], x.shape[1], y.shape[1])
            mask_y_tri = np.tril(torch.ones((y.shape[0], y.shape[1], y.shape[1])))
            #mask = np.concatenate((mask_y, mask_y_tri), axis=1)
            x, _ = self.bert(x)  # input shape:(batch_size, sen_len)
            if torch.cuda.is_available():
                mask = mask_y_tri.cuda()
            y, _ = self.bert(y, attention_mask=mask_y_tri)
            y_pred = self.classify(y)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


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
