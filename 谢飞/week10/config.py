# -*- coding: utf-8 -*-

"""
配置文件
基于BERT的自回归语言模型训练配置
"""

import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

Config = {
    "model_path": MODEL_DIR,
    "train_data_path": os.path.join(DATA_DIR, "train.json"),
    "valid_data_path": os.path.join(DATA_DIR, "valid.json"),
    "input_max_length": 120,  # 输入（content）最大长度
    "output_max_length": 30,   # 输出（title）最大长度
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT推荐使用较小的学习率
    "hidden_size": 768,  # BERT-base的隐藏层大小
    "num_decoder_layers": 6,  # Decoder层数
    "num_heads": 8,  # 多头注意力头数
    "dropout": 0.1,
    "beam_size": 5,  # Beam search的beam大小
    "seed": 42,
    # BERT模型路径
    "bert_path": "hfl/chinese-bert-wwm-ext",  # 哈工大中文BERT（推荐）
    # "bert_path": "bert-base-chinese",  # 标准BERT中文模型
    # "bert_path": "hfl/chinese-roberta-wwm-ext",  # 哈工大中文RoBERTa
    # 训练数据采样：如果设置，每轮只使用部分数据进行训练（用于快速测试）
    "epoch_data_size": None,  # None表示使用全部数据
}


