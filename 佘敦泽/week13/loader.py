# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config

"""
数据加载
"""
tokenize = BertTokenizer.from_pretrained(Config["bert_path"]) # 使用 bert 的词表
# tokenize = BertTokenizer(vocab_file=Config["vocab_path"]) # 为了和model一致, 使用自带的词表

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                # input_ids = self.encode_sentence(sentenece) # 不使用  bert 编码
                input_ids = self.encode_sentence(sentenece) # 使用  bert 编码, 编码截取方式
                labels = self.padding(labels, -1)

                # 使用 bert 模型方式, label 填充方式
                # input_ids = self.encode_sentence_bert_with_cls_sep(sentenece)  # 使用bert 来编码
                # adjusted_labels = [8] + labels + [8]
                # labels = self.padding(adjusted_labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    def encode_sentence_bert(self, text, padding=True): # 使用截取的方式, 如果不调整这里，调整 label 情况怎么样 ?
        text = "".join(text)
        encode_data = tokenize.encode_plus(text, padding=False, truncation=True, max_length=self.config["max_length"], return_length=True)
        input_ids = encode_data["input_ids"][1:-1] # 去掉CLS和SEP

        if padding:
            input_ids = self.padding(input_ids)
        return input_ids

    def encode_sentence_bert_with_cls_sep(self, text, padding=True): # 使用截取的方式, 如果不调整这里，调整 label 情况怎么样 ?
        text = "".join(text)
        encode_data = tokenize.encode_plus(text, padding=False, truncation=True, max_length=self.config["max_length"]-2, return_length=True)
        input_ids = encode_data["input_ids"]

        if padding:
            input_ids = self.padding(input_ids)
        return input_ids

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("./ner_data/train", Config)

