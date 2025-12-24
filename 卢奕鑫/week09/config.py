# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",  # 可以保留或使用BERT的vocab
    "max_length": 100,
    "hidden_size": 768,  # BERT-base 的 hidden_size 是 768
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT通常使用更小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r".\bert-base-chinese",  # 你的BERT路径
    "bert_hidden_size": 768,
    "dropout": 0.1
}

