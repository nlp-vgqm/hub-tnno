# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_tag_news.json",
    "valid_data_path": "data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 128,
    "kernel_size": 6,
    "num_layers": 6,
    "epoch": 20,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"E:\python\学习相关\第六周 语言模型\week6 语言模型和预训练\bert-base-chinese",
    "seed": 987,
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "num_classes": 9
}