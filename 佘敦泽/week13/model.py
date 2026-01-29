# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF # pip install pytorch-crf
from transformers import BertModel
from peft import LoraConfig, get_peft_model  # 导入LoRA相关功能

"""
建立网络模型结构
"""

# 如果使用bert，则使用BertTorchModel
class BertTorchModel(nn.Module):
    def __init__(self, bert_model:BertModel, config):
        super(BertTorchModel, self).__init__()
        
        # 配置LoRA
        if config.get("use_lora", False):
            lora_config = LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["lora_target_modules"],
                lora_dropout=config["lora_dropout"],
                bias="none",
            )
            # 使用LoRA包装BERT模型
            self.bert_model = get_peft_model(bert_model, lora_config)
            # 打印LoRA可训练参数
            # print(self.bert_model.state_dict().keys())
            self.bert_model.print_trainable_parameters()
        else:
            self.bert_model = bert_model
            for param in self.bert_model.parameters():  # 这里冻结 bert 预训练模型的参数
                param.requires_grad_ = False

        self.hidden_size = bert_model.config.hidden_size
        self.vocab_size = bert_model.config.vocab_size # +1 ?
        self.embedding = bert_model.embeddings.word_embeddings

        class_num = config["class_num"]

        self.classify = nn.Linear(self.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # 1. 直接经过 bert 模型 -- NOTE: 注意需要和 __init__ 中的参数对应
        # explicitly specify input_ids to avoid ambiguity and do not pass labels/target to bert_model
        bert_output = self.bert_model(input_ids=x, return_dict=True)
        x = bert_output.last_hidden_state

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

    bert_model = BertModel.from_pretrained(Config["bert_path"])
    model = BertTorchModel(bert_model,Config)