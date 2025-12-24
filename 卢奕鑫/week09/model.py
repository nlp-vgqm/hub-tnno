# model.py
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.config = config

        # 加载BERT模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        # 冻结BERT的部分参数（可选）
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # BERT的隐藏层大小
        bert_hidden_size = self.bert.config.hidden_size

        # 用于NER的分类层
        self.classify = nn.Linear(bert_hidden_size, config["class_num"])
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

        # CRF层
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]

        # 交叉熵损失（用于非CRF模式）
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None, attention_mask=None):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len]

        if attention_mask is None:
            attention_mask = x.gt(0).int()

        # BERT前向传播
        bert_outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 取最后一层的输出
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)

        # 分类层
        predict = self.classify(sequence_output)  # [batch_size, seq_len, class_num]

        if target is not None:
            if self.use_crf:
                # 使用原始的attention_mask
                mask = attention_mask.bool()

                # 关键修复：创建一个修正后的target
                # 将所有-1标签替换为一个有效的标签（比如0，对应"O"标签）
                corrected_target = target.clone()

                # 找到需要修正的位置：mask为True但标签为-1
                # 这包括[CLS]、[SEP]等特殊标记位置
                needs_fix = (corrected_target == -1) & mask

                # 将这些位置的标签设为0（"O"标签，表示非实体）
                # 注意：0必须是你schema.json中"O"标签对应的索引
                # 检查你的schema.json，"O"应该是8，但CRF需要有效的非-1值
                # 我们先用0，如果不行再调整
                corrected_target[needs_fix] = 0  # 或 8，取决于你的标签体系

                print(f"[调试] 需要修正的标签数量: {needs_fix.sum().item()}")
                print(f"[调试] 修正后[CLS]位置标签: {corrected_target[:, 0]}")
                print(f"[调试] Mask[CLS]位置: {mask[:, 0]}")

                # 验证：确保没有mask为True但标签为-1的情况
                invalid = (corrected_target == -1) & mask
                if invalid.any():
                    print(f"[警告] 仍有{invalid.sum().item()}个位置mask为True但标签为-1")
                    # 将这些也修正
                    corrected_target[invalid] = 0

                return -self.crf_layer(predict, corrected_target, mask=mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    # 对BERT参数和其他参数使用不同的学习率
    bert_params = list(model.bert.named_parameters())
    classifier_params = list(model.classify.named_parameters()) + list(model.crf_layer.named_parameters())

    # 分别设置参数组
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        # BERT参数
        {
            "params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
            "lr": learning_rate,
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
            "lr": learning_rate,
            "weight_decay": 0.0
        },
        # 分类器和CRF参数
        {
            "params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
            "lr": learning_rate * 10,  # 通常给分类层更高的学习率
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
            "lr": learning_rate * 10,
            "weight_decay": 0.0
        },
    ]

    if optimizer == "adam":
        return Adam(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(optimizer_grouped_parameters, lr=learning_rate)
