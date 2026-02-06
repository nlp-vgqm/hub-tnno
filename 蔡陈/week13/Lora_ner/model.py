# -*- coding: utf-8 -*-
"""\
模型（LoRA + BERT + CRF NER）

结构：
- Encoder: AutoModel（Transformer）
- Emission: Linear(hidden -> num_labels)
- Decode/Loss: CRF（线性链）

LoRA：
- 使用 PEFT 将 LoRA 注入到 encoder 的注意力投影层（默认 query/key/value）
- 只优化 requires_grad=True 的参数（LoRA + emission/CRF）
"""
from peft import PeftModel

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

from crf import CRF


class TorchModel(nn.Module):
    def __init__(self, config: dict, id2label: dict, label2id: dict):
        super().__init__()
        self.config = config
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = len(label2id)

        # encoder
        self.backbone = AutoModel.from_pretrained(config["pretrained_model"])

        # LoRA 注入
        if config.get("use_lora", True):
            lora_cfg = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.1),
                bias="none",
                target_modules=config.get("lora_target_modules", ["query", "key", "value"]),
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            # 一些模型可能是 d_model
            hidden_size = getattr(self.backbone.config, "d_model")

        # emission layer
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        # CRF
        self.use_crf = bool(config.get("use_crf", True))
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
            self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None, position_ids=None, **kwargs):
        """
        安全版 forward：
        - 只把 encoder 实际需要的键传给 backbone（避免把 labels 或其他无关 kwargs 传入）
        - 处理 CRF / loss / decode
        """
        # --- 1) 只给 backbone 传允许的输入，避免传入 labels 等多余字段 ---
        encoder_inputs = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            encoder_inputs["position_ids"] = position_ids

        # 如果还有其他可能需要的（例如 past_key_values 等），可以在这里显式添加
        # 但不要把 labels、labels_mask、raw_text 等传入 backbone

        # 关键：绕开 PeftModel.forward，直接调用其内部真实模型 forward（LoRA 已经注入在里面）
        encoder = self.backbone
        if isinstance(self.backbone, PeftModel):
            # peft 注入 LoRA 的模块都在 base_model 内部，这里直接拿到真实 encoder
            if hasattr(self.backbone, "get_base_model"):
                encoder = self.backbone.get_base_model()
            elif hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
                encoder = self.backbone.base_model.model

        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 大多数 transformer 的 encoder 输出在 last_hidden_state
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        # mask: attention_mask 为 None 时构造全 True
        mask = None
        if attention_mask is not None:
            # 保证为 bool tensor
            mask = attention_mask.bool()
        else:
            mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        loss = None
        if labels is not None:
            if getattr(self, "use_crf", False):
                # CRF 需要只计算有效位置；labels 中 -100 表示忽略
                valid_mask = mask & (labels != -100)
                # CRF impl 需要没有 -100 的标签张量，所以把 -100 -> 0（值不参与 log-likelihood 计算）
                tags = torch.where(labels == -100, torch.zeros_like(labels), labels)
                log_likelihood = self.crf(emissions, tags, mask=valid_mask, reduction="mean")
                loss = -log_likelihood
            else:
                loss_fct = self.loss_fct if hasattr(self, "loss_fct") else torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))

        # decode -> predictions (word-level aligned to model input length)
        if getattr(self, "use_crf", False):
            paths = self.crf.decode(emissions, mask=mask)
            B, L = input_ids.shape
            pred = torch.zeros((B, L), dtype=torch.long, device=input_ids.device)
            for i, path in enumerate(paths):
                n = min(len(path), L)
                pred[i, :n] = torch.tensor(path[:n], dtype=torch.long, device=input_ids.device)
        else:
            pred = torch.argmax(emissions, dim=-1)

        return {"loss": loss, "emissions": emissions, "predictions": pred}


    def print_trainable_parameters(self):
        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

    def save_head(self, save_dir: str):
        """保存 emission + CRF 参数（LoRA adapter 由 backbone.save_pretrained 保存）"""
        head = {
            "classifier": self.classifier.state_dict(),
            "use_crf": self.use_crf,
            "crf": self.crf.state_dict() if self.crf is not None else None,
        }
        torch.save(head, os.path.join(save_dir, "head.pt"))

    def load_head(self, load_dir: str, map_location: str = "cpu"):
        path = os.path.join(load_dir, "head.pt")
        ckpt = torch.load(path, map_location=map_location)
        self.classifier.load_state_dict(ckpt["classifier"])
        if ckpt.get("use_crf", True):
            if self.crf is None:
                self.crf = CRF(self.num_labels, batch_first=True)
            self.crf.load_state_dict(ckpt["crf"])


def build_optimizer(config: dict, model: nn.Module):
    from torch.optim import AdamW

    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))
