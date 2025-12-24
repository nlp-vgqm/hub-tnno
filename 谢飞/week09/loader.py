# -*- coding: utf-8 -*-

"""
数据加载器
支持 BERT tokenizer 的数据加载
"""

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class NERDataset(Dataset):
    """NER 数据集类"""
    
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.sentences = []
        self.data = []
        self.load()
    
    def load(self):
        """加载数据"""
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue
                
                sentence = []
                labels = []
                
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    char, label = parts[0], parts[1]
                    sentence.append(char)
                    labels.append(self.schema.get(label, self.schema["O"]))
                
                if len(sentence) == 0:
                    continue
                
                # 保存原始句子
                self.sentences.append("".join(sentence))
                
                # 编码句子和标签
                input_ids, attention_mask, labels_encoded = self.encode_sentence(sentence, labels)
                
                self.data.append([
                    torch.LongTensor(input_ids),
                    torch.LongTensor(attention_mask),
                    torch.LongTensor(labels_encoded)
                ])
    
    def encode_sentence(self, sentence, labels):
        """
        使用 BERT tokenizer 编码句子
        注意：BERT 会对某些字符进行子词分割，需要对齐标签
        对于中文 BERT，通常每个字符对应一个 token
        """
        # 将句子转换为字符串
        text = "".join(sentence)
        
        # 使用 BERT tokenizer 编码
        try:
            # 尝试使用 offset_mapping（transformers >= 4.0）
            encoded = self.tokenizer(
                text,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            has_offset_mapping = True
        except:
            # 如果不支持 offset_mapping，使用基本编码
            encoded = self.tokenizer(
                text,
                max_length=self.config["max_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            has_offset_mapping = False
        
        input_ids = encoded["input_ids"].squeeze(0).tolist()
        attention_mask = encoded["attention_mask"].squeeze(0).tolist()
        
        # 对齐标签
        labels_encoded = []
        char_idx = 0
        
        if has_offset_mapping and "offset_mapping" in encoded:
            # 使用 offset_mapping 对齐
            offset_mapping = encoded["offset_mapping"].squeeze(0).tolist()
            for i, (start, end) in enumerate(offset_mapping):
                if i == 0:  # [CLS] token
                    labels_encoded.append(-1)
                    continue
                
                if input_ids[i] == self.tokenizer.sep_token_id or input_ids[i] == self.tokenizer.pad_token_id:
                    labels_encoded.append(-1)
                    continue
                
                # 对于中文，通常每个字符对应一个 token
                if start < len(labels):
                    labels_encoded.append(labels[start])
                else:
                    labels_encoded.append(-1)
        else:
            # 使用 token 判断方法对齐（适用于中文 BERT，通常每个字符一个 token）
            for i, token_id in enumerate(input_ids):
                if i == 0:  # [CLS] token
                    labels_encoded.append(-1)
                    continue
                
                if token_id == self.tokenizer.sep_token_id or token_id == self.tokenizer.pad_token_id:
                    labels_encoded.append(-1)
                    continue
                
                # 获取 token 文本
                token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                
                # 判断是否是子词（以 ## 开头）或特殊 token
                if token.startswith("##") or token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    labels_encoded.append(-1)
                else:
                    # 对于中文 BERT，通常每个字符对应一个 token
                    if char_idx < len(labels):
                        labels_encoded.append(labels[char_idx])
                        char_idx += 1
                    else:
                        labels_encoded.append(-1)
        
        # 确保长度一致
        while len(labels_encoded) < len(input_ids):
            labels_encoded.append(-1)
        
        # 截断到 max_length
        labels_encoded = labels_encoded[:self.config["max_length"]]
        
        return input_ids, attention_mask, labels_encoded
    
    def load_schema(self, path):
        """加载标签 schema"""
        with open(path, encoding="utf8") as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    """
    用torch自带的DataLoader类封装数据
    Args:
        data_path: 数据文件路径
        config: 配置字典
        shuffle: 是否打乱数据
    Returns:
        DataLoader 对象
    """
    dataset = NERDataset(data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda x: (
            (torch.stack([item[0] for item in x]), torch.stack([item[1] for item in x])),  # (input_ids, attention_mask)
            torch.stack([item[2] for item in x])  # labels
        )
    )
    return dataloader


if __name__ == "__main__":
    from config import Config
    loader = load_data(Config["train_data_path"], Config)
    print(f"数据加载成功，共 {len(loader.dataset)} 条样本")

