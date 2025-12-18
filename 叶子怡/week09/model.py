# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]

        self.bert_conf = BertConfig.from_pretrained(config["pretrain_model_path"])
        self.bert_conf.num_hidden_layers = config["num_layers"]
        self.layer = BertModel(self.bert_conf)

        hidden_size = self.layer.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):

        attention_mask = x.gt(0)
        x = self.layer(input_ids=x, attention_mask=attention_mask,return_dict=True)      #input shape:(batch_size, sen_len, hidden_size)
        predict = self.classify(x.last_hidden_state) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean") # pytorch-crf库的结果要取相反数才是loss
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