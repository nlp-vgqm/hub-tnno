# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    # "train_data_path": "../data/train.json",
    "train_data_path": "../data/triplets_data.json",  # 改为三元组数据
    "original_train_data_path": "../data/data.json",  # 原始训练数据路径
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 50,
    "hidden_size": 512,
    "epoch": 50,
    "batch_size": 64,
    "epoch_data_size": 1000,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例（用于数据采样）
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 0.2,  # 三元组损失的margin参数
    "use_triplet_loss": True,  # 是否使用三元组损失
}
