# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging

from peft import LoraConfig, get_peft_model

from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    peft_config = LoraConfig(
        r = 8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
    model = get_peft_model(model, peft_config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # 根据模型类型处理输入
            if hasattr(model, 'bert'):  # 如果是BERT模型
                # 假设batch_data包含: [input_ids, attention_mask, labels]
                if len(batch_data) == 3:
                    input_ids, attention_mask, labels = batch_data
                    loss = model(input_ids, target=labels)  # attention_mask在模型内部生成
                else:
                    # 如果没有attention_mask，模型会自己生成
                    input_ids, labels = batch_data
                    loss = model(input_ids, target=labels)
            else:  # 原始LSTM模型
                input_id, labels = batch_data
                loss = model(input_id, labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
