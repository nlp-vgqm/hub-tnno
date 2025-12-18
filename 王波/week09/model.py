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
        hidden_size = config["hidden_size"]
        max_length = config["max_length"]
        class_num = config["class_num"]

        # 使用BERT预训练模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_config = BertConfig.from_pretrained(config["bert_path"])
        bert_hidden_size = bert_config.hidden_size

        # BERT输出层微调
        self.fc = nn.Linear(bert_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None, attention_mask=None):
        # BERT前向传播
        bert_outputs = self.bert(x, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]  # (batch_size, seq_len, hidden_size)

        # 微调层
        x = self.fc(sequence_output)
        x = self.dropout(x)
        predict = self.classify(x)  # (batch_size, seq_len, num_tags)

        if target is not None:
            if self.use_crf:
                # 创建mask，-1的位置为False，其他为True
                # 注意：我们现在将CLS/SEP标记为8，PAD标记为-1
                mask = target.gt(-1)

                # 确保mask的第一个位置（CLS）是True
                mask[:, 0] = 1

                # 注意这个负号，该库中loss定义与公式相反，则需要取反
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # (number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # 预测时也需要mask
                mask = x.gt(-1)  # 使用x创建mask，但这里需要更精确
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