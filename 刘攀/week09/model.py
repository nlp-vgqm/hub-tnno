# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF  # 确保这是我之前提供的修复版torchcrf.py
from transformers import BertModel, BertConfig

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]  # 提前定义，方便后续使用

        # 修复点1：完整加载BERT预训练模型（配置+权重）
        self.bert_conf = BertConfig.from_pretrained(config["pretrain_model_path"])
        self.bert_conf.num_hidden_layers = config["num_layers"]
        # 加载预训练权重（而非仅加载配置），避免权重随机
        self.layer = BertModel.from_pretrained(
            config["pretrain_model_path"],
            config=self.bert_conf
        )

        hidden_size = self.layer.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)

        # CRF层初始化（batch_first=True，与输入维度对齐）
        if self.use_crf:
            self.crf_layer = CRF(class_num, batch_first=True)

        # 交叉熵损失：ignore_index=-1 忽略padding标签
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # 当输入真实标签，返回loss值；无真实标签，返回预测值

    def forward(self, x, target=None):
        # 生成attention_mask：bool型，[batch_size, sen_len]，过滤padding（x=0）
        attention_mask = x.gt(0)

        # BERT前向计算：last_hidden_state shape=(batch_size, sen_len, hidden_size)
        x_output = self.layer(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 分类头：predict shape=(batch_size, sen_len, class_num)
        predict = self.classify(x_output.last_hidden_state)

        # 训练模式（有target）
        if target is not None:
            if self.use_crf:
                # 生成CRF的mask：bool型，[batch_size, sen_len]，过滤target=-1的无效标签
                crf_mask = target.gt(-1)
                # 调用CRF计算损失（mask为bool型，与CRF要求一致）
                loss = -self.crf_layer(
                    emissions=predict,
                    tags=target,
                    mask=crf_mask,
                    reduction="mean"
                )
                return loss
            else:
                # 非CRF模式：展平计算交叉熵损失
                return self.loss(
                    predict.view(-1, predict.shape[-1]),  # (batch*sen_len, class_num)
                    target.view(-1)  # (batch*sen_len,)
                )
        # 推理模式（无target）
        else:
            if self.use_crf:
                # 修复点2：解码时必须传入mask，过滤padding部分
                preds = self.crf_layer.decode(predict, mask=attention_mask)
                return preds
            else:
                # 非CRF模式：返回每个位置的预测标签（可选返回概率）
                return torch.argmax(predict, dim=-1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    # 模拟Config配置（避免依赖外部config.py，方便测试）
    class Config:
        pretrain_model_path = "bert-base-chinese"  # 可替换为本地BERT路径
        num_layers = 6  # BERT隐藏层数量
        class_num = 5  # 标签数量（根据你的任务调整）
        use_crf = True  # 使用CRF
        optimizer = "adam"
        learning_rate = 1e-5


    # 初始化模型并测试前向传播
    model = TorchModel(Config)
    # 构造测试数据：batch_size=2，sen_len=4
    x = torch.tensor([[101, 2345, 3456, 0], [101, 4567, 5678, 0]])  # 含padding的输入
    target = torch.tensor([[1, 2, 3, -1], [2, 3, 4, -1]])  # 含-1的标签

    # 测试训练模式（计算loss）
    model.train()
    loss = model(x, target)
    print(f"CRF损失值：{loss.item():.4f}")

    # 测试推理模式（预测标签）
    model.eval()
    with torch.no_grad():
        preds = model(x)
    print(f"预测标签序列：{preds}")