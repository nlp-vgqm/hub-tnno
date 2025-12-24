# loader.py
import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path

        # 加载BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.config["vocab_size"] = len(self.tokenizer)

        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])

                text = "".join(sentence)
                self.sentences.append(text)

                # 使用BERT tokenizer编码
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"].squeeze(0)
                attention_mask = encoding["attention_mask"].squeeze(0)

                # 调整labels长度以匹配tokenized的结果
                # 注意：BERT tokenizer可能会将单个字符拆分成多个subword
                # 这里简单处理，使用原始字符的label（实际使用时可能需要更精细的处理）
                aligned_labels = self.align_labels(sentence, labels, input_ids)

                self.data.append([
                    input_ids,
                    torch.LongTensor(aligned_labels),
                    attention_mask
                ])
        return

    def align_labels(self, sentence, labels, input_ids):
        """更保守的对齐方法"""
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        aligned_labels = [-1] * len(tokens)  # 初始化为-1

        char_index = 0
        for i, token in enumerate(tokens):
            if token == "[PAD]":
                continue  # 保持-1
            elif token == "[CLS]" or token == "[SEP]":
                aligned_labels[i] = -1  # 明确设为-1
                continue
            elif token.startswith("##"):
                # subword：如果前一个token有有效标签，则延续
                if i > 0 and aligned_labels[i - 1] != -1:
                    aligned_labels[i] = aligned_labels[i - 1]
            else:
                # 常规token
                if char_index < len(labels):
                    aligned_labels[i] = labels[char_index]
                    char_index += 1

        return aligned_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
