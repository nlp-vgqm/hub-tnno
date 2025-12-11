# -*- coding: utf-8 -*-

"""
配置文件
所有路径相对于当前文件所在目录
"""

import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置（相对于当前目录）
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

Config = {
    "model_path": MODEL_DIR,
    "schema_path": os.path.join(DATA_DIR, "schema.json"),
    "train_data_path": os.path.join(DATA_DIR, "train.json"),
    "valid_data_path": os.path.join(DATA_DIR, "valid.json"),
    "vocab_path": os.path.join(BASE_DIR, "chars.txt"),
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,      # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例（三元组损失不需要，保留用于兼容）
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 0.5,               # 三元组损失的margin参数
    "distance_type": "euclidean", # 距离类型: 'euclidean' 或 'cosine'
}

