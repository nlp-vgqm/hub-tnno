# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path

        # 使用BERT的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
        if not os.path.exists(self.path):
            print(f"错误: 文件不存在: {self.path}")
            return

        with open(self.path, encoding="utf8") as f:
            content = f.read()
            segments = content.split("\n\n")
            print(f"找到 {len(segments)} 个句子段")

            for segment_idx, segment in enumerate(segments):
                if segment.strip() == "":
                    continue

                original_chars = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        char, label = parts[0], parts[-1]
                        original_chars.append(char)
                        labels.append(self.schema[label])

                # 原始句子
                sentence = "".join(original_chars)
                self.sentences.append(sentence)

                # 使用BERT tokenizer编码
                encoding = self.tokenizer(
                    sentence,
                    padding="max_length",
                    truncation=True,
                    max_length=self.config["max_length"],
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].squeeze(0)
                attention_mask = encoding["attention_mask"].squeeze(0)

                # 对齐标签
                aligned_labels = self.align_labels_to_tokens(original_chars, labels, encoding)

                self.data.append([
                    input_ids,
                    torch.LongTensor(aligned_labels),
                    attention_mask
                ])

                # 显示前几个样本的信息
                if segment_idx < 2:
                    print(f"样本 {segment_idx + 1}:")
                    print(f"  原始句子: {sentence[:50]}...")
                    print(f"  标签数量: {len(labels)}")
                    print(f"  对齐后标签数量: {len(aligned_labels)}")
                    print(f"  实体标签数量: {sum(1 for l in labels if l < 8)}")

        print(f"数据加载完成，共 {len(self.data)} 个样本")
        return

    def align_labels_to_tokens(self, original_chars, original_labels, encoding):
        """
        将原始字符级别的标签对齐到BERT的tokenized序列
        """
        # 获取tokenized的tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze(0).tolist())

        aligned_labels = []
        char_idx = 0

        for i, token in enumerate(tokens):
            if i >= self.config["max_length"]:
                break

            if token in ["[PAD]", "[CLS]", "[SEP]"]:
                # 特殊标记对应标签-1（padding）或8（CLS/SEP）
                if token == "[PAD]":
                    aligned_labels.append(-1)
                else:
                    # CLS和SEP标记对应标签8（O标签）
                    aligned_labels.append(8)  # 8对应'O'标签
                continue

            # 处理子词标记
            if token.startswith("##"):
                # 子词标记，继承前一个token的标签
                if aligned_labels:
                    prev_label = aligned_labels[-1]
                    # 如果是B-标签，则子词转为I-标签
                    if prev_label in [0, 1, 2, 3]:  # B-标签
                        aligned_labels.append(prev_label + 4)  # 转为对应的I-标签
                    else:
                        aligned_labels.append(prev_label)
                else:
                    aligned_labels.append(8)  # 默认O标签
            else:
                # 正常token，对齐到原始字符
                if char_idx < len(original_labels):
                    aligned_labels.append(original_labels[char_idx])
                    char_idx += 1
                else:
                    # 这种情况不应该发生，但如果发生了，给一个默认标签
                    aligned_labels.append(8)

        # 确保长度一致
        while len(aligned_labels) < self.config["max_length"]:
            aligned_labels.append(-1)  # padding位置用-1

        return aligned_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("ner_data/train", Config)