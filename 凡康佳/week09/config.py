# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "vocab_size": 5000,
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 8,  # 减小batch size，BERT通常需要较小batch
    "optimizer": "adam",
    "learning_rate": 3e-5,  # BERT微调使用更小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"E:\python\学习相关\第六周 语言模型\week6 语言模型和预训练\bert-base-chinese"
}

