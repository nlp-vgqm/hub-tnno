#coding:utf8
from os import truncate

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import json
import re
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import Dataset, DataLoader



"""
用bert+mask实现sft的效果
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_dim)

        self.config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=input_dim,
            num_hidden_layers=2,
            num_attention_heads=4,
            return_dict = True
        )
        self.bert = BertModel(self.config)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, token_type_ids=None):
        # print(input_ids.shape)
        if token_type_ids is not None:
            y = self.get_y(input_ids)
            mask = self.create_mask(input_ids, token_type_ids)
            input_ids = self.bert(input_ids, mask)
            y_pred = self.classify(input_ids.last_hidden_state)  # output shape:(batch_size, sen_len, vocab_size)
            return self.cal_loss(y_pred, y, token_type_ids)
        else:
            input_ids = self.bert(input_ids)
            y_pred = self.classify(input_ids.last_hidden_state)   #output shape:(batch_size, sen_len, vocab_size)
            return torch.softmax(y_pred, dim=-1)

    def cal_loss(self, y_pred, y, token_type_ids):
        batch_size, seq_len = y.shape
        total_loss = 0
        for b in range(batch_size):
            _, s1_end, s2_start, s2_end = self.get_boundry(token_type_ids[b])
            batch_y_pred  = y_pred[b, s1_end:s2_end, :]
            batch_y = y[b, s1_end:s2_end]
            batch_loss = self.loss(batch_y_pred.view(-1, batch_y_pred.shape[-1]), batch_y.view(-1))
            total_loss += batch_loss
        return total_loss


    def get_boundry(self, single_token_type_ids):
        s2_pos = (single_token_type_ids == 1).nonzero().squeeze().tolist()
        s1_start = 0
        s2_start = -1
        s2_end = -1
        if s2_pos:
            if isinstance(s2_pos, int):
                s2_start = s2_pos
                s2_end = s2_pos
            else:
                s2_start = min(s2_pos)
                s2_end = max(s2_pos)
        s1_end = s2_start - 1
        return s1_start, s1_end, s2_start, s2_end

    def get_y(self, input_ids):
        batch_size, seq_len = input_ids.shape
        y = torch.zeros(batch_size, seq_len, dtype=input_ids.dtype)
        y[:, :-1] = input_ids[:, 1:]
        y[:, -1] = 102
        return y

    def create_mask(self, input_ids, token_type_ids):
        batch_size, seq_len = input_ids.shape

        batch_sft_mask = torch.zeros(batch_size, seq_len, seq_len)
        batch_padding_mask = torch.zeros(batch_size, seq_len, seq_len)
        for b in range(batch_size):
            _, s1_end, s2_start, s2_end = self.get_boundry(token_type_ids[b])
            batch_sft_mask[b, :s1_end+1, :s1_end+1] = 1
            batch_sft_mask[b, s2_start:, :s1_end+1] = 1
            for i in range(s2_start, seq_len):
                batch_sft_mask[b, i, :i+1] = 1
                batch_sft_mask[b, i, i+1:] = 0

            batch_padding_mask[b, :s2_end+1, :s2_end+1] =1
        mask = batch_sft_mask.bool() & batch_padding_mask.bool()
        return mask


#加载字表
def build_vocab(pretrain_path):
    return BertTokenizer.from_pretrained(pretrain_path)


class DataGenerator:
    def __init__(self, tokenizer, data_path, max_len):
        self.tokenizer = tokenizer
        self.s1, self.s2 = load_corpus(data_path)
        self.max_len = max_len

        self.data = []
        self.load()

    def load(self):
        for s1, s2 in zip(self.s1, self.s2):
            encoding = self.tokenizer(s1, s2, padding='max_length', return_tensors='pt',
                                      truncation="only_second", max_length=self.max_len)
            self.data.append([encoding['input_ids'].squeeze(0), encoding['token_type_ids'].squeeze(0)])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#用torch自带的DataLoader类封装数据
def load_data(tokenizer, data_path, max_len, batch_size=10, shuffle=True):
    dg = DataGenerator(tokenizer, data_path, max_len)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl

#加载语料
def load_corpus(path):
    s1 = []
    s2 = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            s1.append(title)
            s2.append(content)
    return s1, s2


#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]  # 取倒数10个字
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            # print(y.shape, y)
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution) # 根据prob_distribution的概率来对其采样


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 50        #训练轮数
    batch_size = 10       #每次训练样本个数
    char_dim = 256        #每个字的维度
    pretrain_path = r"D:\pretrain_models\bert-base-chinese"
    tokenizer = build_vocab(pretrain_path)       #建立字表
    vocab = tokenizer.vocab    #加载语料
    max_len = 100

    # 加载训练数据
    train_data = load_data(tokenizer, corpus_path, max_len, batch_size)

    model = build_model(vocab, char_dim)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3 )  #建立优化器
    print("文本词表模型加载完毕，开始训练")


    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for index, batch_data in enumerate(train_data):

            input_ids, token_type_ids = batch_data
            if torch.cuda.is_available():
                input_ids, token_type_ids = input_ids.cuda(), token_type_ids.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(input_ids, token_type_ids)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("互联网要有社会担当", model, vocab, 10))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, vocab, 10))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
