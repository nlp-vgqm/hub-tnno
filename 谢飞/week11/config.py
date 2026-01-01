# -*- coding: utf-8 -*-

"""
配置文件
SFT (Supervised Fine-Tuning) 式 seq2seq 训练配置
"""

import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")

Config = {
    "model_path": MODEL_DIR,
    "train_data_path": os.path.join(DATA_DIR, "train.json"),
    "valid_data_path": os.path.join(DATA_DIR, "valid.json"),
    "max_length": 512,  # 最大序列长度（input + output）
    "epoch": 5,
    "batch_size": 4,  # SFT通常使用较小的batch size
    "optimizer": "adamw",
    "learning_rate": 2e-5,  # SFT推荐使用较小的学习率
    "weight_decay": 0.01,  # L2正则化
    "warmup_steps": 100,  # 学习率预热步数
    "gradient_accumulation_steps": 4,  # 梯度累积步数（模拟更大的batch size）
    "max_grad_norm": 1.0,  # 梯度裁剪
    "save_steps": 500,  # 每多少步保存一次模型
    "eval_steps": 200,  # 每多少步评估一次
    "seed": 42,
    # 预训练模型路径（支持GPT、T5等生成模型）
    # 推荐使用中文生成模型：
    "model_name": "gpt2",  # 使用GPT-2作为基础模型
    # "model_name": "uer/gpt2-chinese-cluecorpussmall",  # 中文GPT-2
    # "model_name": "THUDM/chatglm-6b",  # ChatGLM（需要较大显存）
    # "model_name": "hfl/chinese-llama-2-7b",  # Chinese LLaMA（需要较大显存）
    # 训练数据采样：如果设置，每轮只使用部分数据进行训练（用于快速测试）
    "epoch_data_size": None,  # None表示使用全部数据
    # 数据格式：支持两种格式
    # "instruction-response": {"instruction": "...", "response": "..."}
    # "input-output": {"input": "...", "output": "..."}
    "data_format": "instruction-response",  # 或 "input-output"
}


