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
        #hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        #self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        #self.use_bert = True
        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], num_hidden_layers=6,return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        #print(f'{hidden_size=}')
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值

    def forward(self, x, target=None):
        #x = self.embedding(x)  #input shape:(batch_size, sen_len)
        output, pooling_output = self.encoder(x)    #output shape:(batch_size, sen_len, dim)
        #print(x.shape)
        predict = self.classify(output) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))     #输入为[batchsize*len(sen),dim],输出为1个数
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

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"], output_attentions=False,return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        #print(x.shape,'red')
        y = self.bert(x)[0]
        y,_ = self.rnn(y)
        #print(y.shape,'blue')
        return y




if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 71465
    model = TorchModel(Config)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])   #2*5
    y = torch.LongTensor([1,2,3,1,2,3,1,2,3,1])    #1*10
    print(model(x,y),model(x,y).shape)
