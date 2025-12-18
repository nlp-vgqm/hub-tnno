# -*- coding: utf-8 -*-

"""
基于 BERT 的 NER 模型定义
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer
from torchcrf import CRF


class BertNERModel(nn.Module):
    """基于 BERT 的命名实体识别模型"""
    
    def __init__(self, config):
        super(BertNERModel, self).__init__()
        self.config = config
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        
        # 加载 BERT 模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_hidden_size = self.bert.config.hidden_size
        
        # 分类层：将 BERT 输出映射到标签类别
        self.classify = nn.Linear(bert_hidden_size, class_num)
        
        # CRF 层（可选）
        if self.use_crf:
            self.crf_layer = CRF(class_num, batch_first=True)
        
        # 交叉熵损失（不使用 CRF 时）
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Dropout 层
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, target=None):
        """
        前向传播
        当输入真实标签，返回loss值；无真实标签，返回预测值
        Args:
            x: 输入数据，可以是 input_ids 或 (input_ids, attention_mask) 元组
            target: 真实标签 (batch_size, seq_len)，可选
        Returns:
            如果提供了 target，返回损失值
            否则返回预测结果
        """
        # 处理输入：支持两种格式
        if isinstance(x, (list, tuple)) and len(x) == 2:
            input_ids, attention_mask = x
        else:
            input_ids = x
            attention_mask = None
        
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类层
        predict = self.classify(sequence_output)  # (batch_size, seq_len, class_num)
        
        if target is not None:
            # 训练模式：计算损失
            if self.use_crf:
                # 创建 mask：优先使用 attention_mask，否则从 target 创建（-1 表示 padding）
                if attention_mask is not None:
                    mask = attention_mask.bool()
                else:
                    mask = target.gt(-1)
                # CRF 要求第一个时间步的 mask 必须全部为 True
                mask[:, 0] = True
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 使用交叉熵损失
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            # 预测模式：返回预测结果
            if self.use_crf:
                if attention_mask is not None:
                    mask = attention_mask.bool()
                else:
                    mask = torch.ones_like(input_ids, dtype=torch.bool)
                # CRF 要求第一个时间步的 mask 必须全部为 True
                mask[:, 0] = True
                try:
                    return self.crf_layer.decode(predict, mask)
                except TypeError:
                    # 如果 decode 不支持 mask 参数，尝试不使用 mask
                    return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支持的优化器: {optimizer}")


if __name__ == "__main__":
    from config import Config
    model = BertNERModel(Config)
    print("模型创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")


