# -*- coding: utf-8 -*-
"""
    加载数据集，做预处理，为训练做准备
"""

import csv
import jieba
import torch

from config import *
from sklearn.model_selection import train_test_split
from model import *


#将数据集按8：2随机分成训练集和测试集
def datacsv(data_path, test_size, random_state, shuffle):
    vocab = {}
    with open(data_path, 'r+', encoding='utf-8-sig') as file:
        reads = csv.reader(file)
        for i, read in enumerate(reads):
            if i > 0:
                vocab[read[1]] = read[0]
    keys = list(vocab.keys())
    values = list(vocab.values())

    train_keys, test_keys, train_values, test_values = train_test_split(
        keys,
        values,
        test_size=test_size,  #数据集20%生成测试集
        random_state=random_state,  #随机种子保证随机生成测试集的可重复性
        shuffle=shuffle  #随机生成测试集时是否可重复
    )
    train_vocab = dict(zip(train_keys, train_values))
    test_vocab = dict(zip(test_keys, test_values))
    return train_vocab, test_vocab


#使用jieba分词将数据集中的文本转换成词表
def text_to_vocab(texts):
    vocab = {"[pad]": 0}
    for text in texts:
        seg_list = list(jieba.cut(text, cut_all=False))
        for i in seg_list:
            if i in vocab:
                pass
            else:
                vocab[i] = len(vocab)
    vocab["[unk]"] = len(vocab)
    return vocab


#建立rnn模型
def model_rnn(output_size, sentence_length, input_size, pooling):
    model = TorchModel_Rnn(output_size, sentence_length, input_size, pooling)
    return model

#建立cnn模型
def model_cnn(output_size, sentence_length, input_size, pooling):
    model = TorchModel_Cnn(output_size, sentence_length, input_size, pooling)
    return model

#建立lstm模型
def model_lstm(output_size, sentence_length, input_size, pooling):
    model = TorchModel_Lstm(output_size, sentence_length, input_size, pooling)
    return model

#建立BERT模型
def model_bert(output_size, sentence_length, input_size):
    model = TorchModel_BERT(output_size, sentence_length, input_size)
    return model

#样本文本长度，按最大长度60%
def text_long(texts):
    num = []
    for i in texts:
        num.append(len(i) - 1)
    sentence_length = int(max(num)*0.6)
    return sentence_length


#创建训练数据
def datatrain_rnn(train_vocab,  vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for text, label in train_vocab.items():
        x = list(jieba.cut(text, cut_all=False))
        x = [word.strip() for word in x if word.strip()]
        if len(x) < sentence_length:
            padding = ['[pad]'] * (sentence_length - len(x))
            x += padding
        elif len(x) > sentence_length:
            x = x[:sentence_length]
        x = [vocab.get(word, vocab['[unk]']) for word in x]
        dataset_x.append(x)
        dataset_y.append([float(label)])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


if __name__ == '__main__':
    train_vocab, test_vocab = datacsv(config['data_path'], config['test_size'], config['random_state'],
                                      config['shuffle'])
    texts = []
    for i in train_vocab.keys():
        texts.append(i)
    text_long(texts)
    x, y = datatrain_rnn(train_vocab, text_to_vocab(texts), text_long(texts))
    print(f"x:{x}, y:{y}")
