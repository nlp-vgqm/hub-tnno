# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例（仅用于二元组训练）
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_triplet": True,        # 是否使用三元组训练，False则使用二元组训练
    "triplet_margin": 0.1,      # 三元组损失中的margin参数
}
