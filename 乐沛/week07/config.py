# -*- coding: utf-8 -*-

"""
    配置参数信息
"""

config = {
    "data_path": "文本分类练习.csv",
    "test_size": 0.2,  #数据集20%生成测试集
    "random_state": 42,  #随机种子保证随机生成测试集的可重复性
    "shuffle": True,  #随机生成测试集时是否可重复
    "epoch_num": 10,    #训练轮数
    "learning_num": 0.005,   #学习率
    "batch_size": 20,  # 每次训练样本个数
    "train_sample": 500,  # 每轮训练总共训练的样本总数
    "char_dim": 20,  # 每个字的维度
}
