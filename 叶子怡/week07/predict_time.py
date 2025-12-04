# -*- coding: utf-8 -*-
import torch
from loader import load_data
import pandas as pd
from transformers import BertTokenizer
import time
from config import Config
from model import TorchModel
"""
模型效果测试
"""

class Predict:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.input_data = []
        self.predict_data = []
        self.text = []
        self.predict()

    def predict(self):

        # 加载模型参数
        model = TorchModel(self.config)
        # 标识是否使用gpu
        cuda_flag = torch.cuda.is_available()
        if cuda_flag:
            model = model.cuda()

        model.load_state_dict(torch.load("output\{model_type}.pth".format(model_type=self.config["model_type"])))
        model.eval()
        # 加载100条数据
        self.csv_data = pd.read_csv(self.data_path).head(100)
        for _, row in self.csv_data.iterrows():
            text = str(row['review'])
            label = int(row['label'])
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(text,
                                                max_length=self.config["max_length"],
                                                truncation=True,  # 明确启用截断
                                                padding='max_length')  # 使用新的padding参数
            else:
                input_id = self.encode_sentence(text)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])

            self.input_data.append(input_id)
            self.predict_data.append(label_index)
            self.text.append(text)


        x = torch.stack(self.input_data)
        y = torch.stack(self.predict_data)
        print(x.shape)
        start_time = time.time()
        with torch.no_grad():
            pred_results = model(x) #不输入labels，使用模型当前参数进行预测
            print(pred_results.shape)
            # for y_p, y_t, text in zip(pred_results, y, self.text):
            #     print(text, "预测结果", y_p, "真实结果", y_t)

        end_time = time.time()
        total_time = end_time - start_time
        print(self.config["model_type"], "预测耗时：",total_time)
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

if __name__ == "__main__":
    # main(Config)
    # logger.disabled = True
    #加载模型
    for model in ["rnn", "lstm", "gated_cnn", "bert", "bert_lstm"]:
        Config["model_type"] = model
        predict = Predict(Config, "data\\test.csv")