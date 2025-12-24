# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载
"""

# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
数据加载 - 支持三元组格式
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]
        self.data_type = None
        self.triplet_data = []  # 存储三元组数据
        self.knwb = defaultdict(list)  # 知识库
        self.data = []  # 测试集数据
        self.load()

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)

                # 检查是否为三元组格式
                if "anchor" in line and "positive" in line and "negative" in line:
                    self.data_type = "train"

                    # 编码三元组
                    anchor_id = self.encode_sentence(line["anchor"])
                    positive_id = self.encode_sentence(line["positive"])
                    negative_id = self.encode_sentence(line["negative"])

                    self.triplet_data.append([
                        torch.LongTensor(anchor_id),
                        torch.LongTensor(positive_id),
                        torch.LongTensor(negative_id)
                    ])

                # 原有的格式（向后兼容）
                elif isinstance(line, dict) and "questions" in line:
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]

                    # 记录到知识库
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)

                # 测试集格式
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

        # 如果是训练数据且没有三元组数据，则从知识库生成
        if self.data_type == "train" and not self.triplet_data and self.knwb:
            self.generate_triplets_from_knwb()

        return

    def generate_triplets_from_knwb(self, num_triplets_per_question=3):
        """从知识库生成三元组数据"""
        standard_question_index = list(self.knwb.keys())

        for category_idx in standard_question_index:
            questions = self.knwb[category_idx]

            if len(questions) < 2:
                continue

            for anchor in questions:
                # 选择正例（同一类别但不同的问题）
                positive_candidates = [q for q in questions if not torch.equal(q, anchor)]
                if not positive_candidates:
                    continue

                for _ in range(num_triplets_per_question):
                    positive = random.choice(positive_candidates)

                    # 选择负例类别
                    negative_categories = [c for c in standard_question_index if c != category_idx and self.knwb[c]]
                    if not negative_categories:
                        continue

                    negative_category = random.choice(negative_categories)
                    negative = random.choice(self.knwb[negative_category])

                    self.triplet_data.append([anchor, positive, negative])

        print(f"从知识库生成了 {len(self.triplet_data)} 个三元组")

    def __len__(self):
        if self.data_type == "train":
            return min(len(self.triplet_data), self.config["epoch_data_size"])
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            if index < len(self.triplet_data):
                return self.triplet_data[index]
            else:
                # 如果索引超出范围，返回随机三元组
                return self.random_triplet_sample()
        else:
            return self.data[index]

    def random_triplet_sample(self):
        """实时生成随机三元组"""
        standard_question_index = list(self.knwb.keys())

        # 随机选择一个类别作为正例类别
        p = random.choice(standard_question_index)
        if len(self.knwb[p]) < 2:
            return self.random_triplet_sample()

        # 从正例类别中随机选择两个问题作为anchor和positive
        anchor, positive = random.sample(self.knwb[p], 2)

        # 随机选择一个不同的类别作为负例类别
        n = random.choice([idx for idx in standard_question_index if idx != p])
        negative = random.choice(self.knwb[n])

        return [anchor, positive, negative]


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

