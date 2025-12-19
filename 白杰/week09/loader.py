# -*- coding: utf-8 -*-

import json
import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载 - 适配BERT
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.id2schema = {v: k for k, v in self.schema.items()}
        self.sentences = []
        self.load()

    def load(self):
        self.data = []
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
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(label)

                self.sentences.append("".join(sentence))
                # 处理标签与token的对齐问题
                input_ids, input_labels = self.process_sentence(sentence, labels)
                self.data.append([
                    torch.LongTensor(input_ids["input_ids"]),
                    torch.LongTensor(input_ids["attention_mask"]),
                    torch.LongTensor(input_ids["token_type_ids"]),
                    torch.LongTensor(input_labels)
                ])
        return

    def process_sentence(self, sentence, labels):
        """处理句子和标签，考虑BERT的分词方式"""
        tokenized_input = self.tokenizer(
            sentence,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 处理标签，考虑子词的情况
        word_ids = tokenized_input.word_ids(batch_index=0)
        input_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS]或[SEP]或padding
                input_labels.append(-1)
            elif word_idx != previous_word_idx:
                # 新词开始
                input_labels.append(self.schema[labels[word_idx]])
            else:
                # 子词，使用I-标签
                label = labels[word_idx]
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                input_labels.append(self.schema.get(label, self.schema["O"]))
            previous_word_idx = word_idx

        return {
            "input_ids": tokenized_input["input_ids"].squeeze().tolist(),
            "attention_mask": tokenized_input["attention_mask"].squeeze().tolist(),
            "token_type_ids": tokenized_input["token_type_ids"].squeeze().tolist()
        }, input_labels

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

    dg = DataGenerator("../ner_data/train.txt", Config)