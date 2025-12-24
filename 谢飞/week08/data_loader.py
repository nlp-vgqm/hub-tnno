# -*- coding: utf-8 -*-

"""
数据加载模块
包含三元组数据生成器和数据加载函数
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def load_vocab(vocab_path):
    """加载字表或词表"""
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_schema(schema_path):
    """加载schema"""
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


class TripletDataGenerator(Dataset):
    """三元组数据生成器"""
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.load()

    def load(self):
        """加载数据"""
        self.data = []
        self.knwb = defaultdict(list)
        
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        """编码句子为ID序列"""
        input_id = []
        if self.config["vocab_path"].endswith("words.txt"):
            try:
                import jieba
                for word in jieba.cut(text):
                    input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
            except ImportError:
                # 如果没有jieba，按字符处理
                for char in text:
                    input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        """补齐或截断序列"""
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.train_data_size
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_triplet_sample()
        else:
            return self.data[index]

    def random_triplet_sample(self):
        """随机生成一个三元组样本 (anchor, positive, negative)"""
        standard_question_index = list(self.knwb.keys())
        
        # 随机选择一个类别作为anchor和positive的类别
        anchor_class = random.choice(standard_question_index)
        
        # 确保该类至少有2个样本（一个作为anchor，一个作为positive）
        if len(self.knwb[anchor_class]) < 2:
            return self.random_triplet_sample()
        
        # 从同一类别中随机选择anchor和positive
        anchor, positive = random.sample(self.knwb[anchor_class], 2)
        
        # 从不同类别中随机选择negative
        negative_class = random.choice([c for c in standard_question_index if c != anchor_class])
        negative = random.choice(self.knwb[negative_class])
        
        return [anchor, positive, negative]


def load_triplet_data(data_path, config, shuffle=True):
    """加载三元组数据"""
    dg = TripletDataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

