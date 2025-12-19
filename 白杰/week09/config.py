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
    "max_length": 128,  # BERT通常使用128或256
    "hidden_size": 768,  # BERT-base的隐藏层维度
    "num_layers": 1,     # 减少额外层数量，因为BERT已经很强大
    "epoch": 30,         # 增加epoch，BERT需要更多训练
    "batch_size": 32,    # 适当增大batch size
    "optimizer": "adamw",# 使用AdamW优化器
    "learning_rate": 2e-5, # BERT微调通常使用较小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\bert-base-chinese",
    "warmup_steps": 100, # 学习率预热步骤
    "weight_decay": 0.01, # 权重衰减，防止过拟合
    "dropout": 0.1       # dropout率
}