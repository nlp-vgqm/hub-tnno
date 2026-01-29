# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"D:\BaiduNetdiskDownload\第九周 序列标注\week9 序列标注问题\week9 序列标注问题\ner\ner_data/schema.json",
    "train_data_path": r"D:\BaiduNetdiskDownload\第九周 序列标注\week9 序列标注问题\week9 序列标注问题\ner\ner_data/train",
    "valid_data_path": r"D:\BaiduNetdiskDownload\第九周 序列标注\week9 序列标注问题\week9 序列标注问题\ner\ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 384,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "use_crf": False,
    "class_num": 9,
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese\bert-base-chinese"
}

