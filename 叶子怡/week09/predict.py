# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from model import TorchModel
import jieba
from config import Config
import json

"""
模型效果测试
"""

class Predictor:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            input_id = [101] + input_id + [102]
            res = self.model(torch.LongTensor([input_id]))[0]
            print("预测结果：", len(res))



if __name__ == "__main__":
    sl = Predictor(Config, "model_output/epoch_5.pth")

    sentence = "据新华社上海2月16日电(记者郭礼华)首届远东俱乐部足球锦标赛将于明晚在新建的上海闸北体育场燃起战火。"
    # bilstm-crf: [8, 1, 5, 5, 0, 4, 3, 7, 7, 7, 7, 8, 8, 8, 8, 2, 6, 6, 8, 8, 8, 0, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 7, 8, 8, 8, 8, 0, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8]

    print("sentence长度：", len(sentence))
    sl.predict(sentence)




