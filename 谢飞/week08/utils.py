# -*- coding: utf-8 -*-

"""
工具函数模块
"""

import torch


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

