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
    "train_data_path": os.path.join(DATA_DIR, "train"),
    "valid_data_path": os.path.join(DATA_DIR, "test"),
    "vocab_path": os.path.join(BASE_DIR, "chars.txt"),
    "max_length": 100,
    "hidden_size": 768,  # BERT-base 的隐藏层大小
    "epoch": 5,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT 通常使用较小的学习率
    "use_crf": True,  # 是否使用 CRF 层
    "class_num": 9,  # 标签类别数（B-LOCATION, B-ORGANIZATION, B-PERSON, B-TIME, I-LOCATION, I-ORGANIZATION, I-PERSON, I-TIME, O）
    # BERT 模型路径：可以是本地路径或 Hugging Face 模型名称
    # 推荐的中文BERT模型选项：
    # - "bert-base-chinese"：标准BERT中文模型（102M参数，较慢但效果好）
    # - "hfl/chinese-bert-wwm-ext"：哈工大中文BERT（全词掩码，效果更好）
    # - "hfl/chinese-roberta-wwm-ext"：哈工大中文RoBERTa（效果更好）
    # - "distilbert-base-chinese"：轻量级BERT（参数量更少，速度更快）
    # 如果使用本地路径，请确保路径存在且包含完整的 BERT 模型文件
    "bert_path": "hfl/chinese-bert-wwm-ext",  # 使用 Hugging Face 模型，会自动下载
    # "bert_path": "hfl/chinese-bert-wwm-ext",  # 哈工大中文BERT（推荐）
    # "bert_path": "hfl/chinese-roberta-wwm-ext",  # 哈工大中文RoBERTa（推荐）
    # "bert_path": r"E:\pretrain_models\bert-base-chinese",  # 或使用本地路径
    # 训练数据采样：如果设置，每轮只使用部分数据进行训练（用于快速测试）
    # None 表示使用全部数据
    # 设置数字如 500 表示每轮只用 500 条样本（可以加快训练速度，适合快速测试）
    "epoch_data_size": 500,
    # 加快训练速度的建议：
    # 1. 使用更小的模型：将 bert_path 改为 "distilbert-base-chinese"（参数量更少）
    # 2. 减少每轮训练数据：设置 epoch_data_size = 500（每轮只用 500 条样本）
    # 3. 减少训练轮数：设置 epoch = 5
    # 4. 使用 GPU：如果有 GPU，程序会自动使用（比 CPU 快很多）
}


