# -*- coding: utf-8 -*-

"""
配置文件
使用LoRA训练NER任务的配置
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
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-4,  # LoRA可以使用稍大的学习率
    "use_crf": True,  # 是否使用 CRF 层
    "class_num": 9,  # 标签类别数（B-LOCATION, B-ORGANIZATION, B-PERSON, B-TIME, I-LOCATION, I-ORGANIZATION, I-PERSON, I-TIME, O）
    
    # BERT 模型路径：可以是本地路径或 Hugging Face 模型名称
    "bert_path": "hfl/chinese-bert-wwm-ext",  # 哈工大中文BERT（推荐）
    # "bert_path": "bert-base-chinese",  # 标准BERT中文模型
    # "bert_path": "hfl/chinese-roberta-wwm-ext",  # 哈工大中文RoBERTa
    
    # LoRA 配置参数
    "lora_r": 8,  # LoRA的秩（rank），控制适配器的大小，越小参数越少
    "lora_alpha": 16,  # LoRA的缩放参数，通常设置为r的2倍
    "lora_dropout": 0.1,  # LoRA的dropout率
    "lora_target_modules": ["query", "key", "value"],  # 要应用LoRA的模块，通常对attention层应用
    # 可选的其他模块：["query", "key", "value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    
    # 训练数据采样：如果设置，每轮只使用部分数据进行训练（用于快速测试）
    "epoch_data_size": None,  # None表示使用全部数据
}
