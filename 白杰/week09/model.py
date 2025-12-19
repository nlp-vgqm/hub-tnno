# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel

"""
基于BERT的网络模型结构
"""


class BertNER(BertPreTrainedModel):
    def __init__(self, config, model_config):
        super(BertNER, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(model_config["dropout"])
        self.classifier = nn.Linear(config.hidden_size, model_config["class_num"])
        self.crf_layer = CRF(model_config["class_num"], batch_first=True)
        self.use_crf = model_config["use_crf"]
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 取最后一层的隐藏状态
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.use_crf:
                # 计算CRF损失
                mask = attention_mask.byte()
                return -self.crf_layer(logits, labels, mask, reduction='mean')
            else:
                # 交叉熵损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                # 只计算有标注部分的损失
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            if self.use_crf:
                # 使用CRF解码
                return self.crf_layer.decode(logits, attention_mask.byte())
            else:
                # 直接取最大概率
                return torch.argmax(logits, dim=2)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    # 区分BERT参数和其他参数，使用不同的学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config["weight_decay"]
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    if optimizer == "adam":
        return Adam(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "adamw":
        return AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)


if __name__ == "__main__":
    from config import Config
    from transformers import BertConfig

    bert_config = BertConfig.from_pretrained(Config["bert_path"])
    model = BertNER(bert_config, Config)