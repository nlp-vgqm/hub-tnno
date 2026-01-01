# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"D:\BaiduNetdiskDownload\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题\data/schema.json",
    "train_data_path": r"D:\BaiduNetdiskDownload\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题\data/train.json",
    "valid_data_path": r"D:\BaiduNetdiskDownload\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题\data/valid.json",
    "vocab_path":"../chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-3,
}
