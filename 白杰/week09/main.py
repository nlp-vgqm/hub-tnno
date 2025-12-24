# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import BertNER, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from transformers import BertConfig, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def set_seed(seed=42):
    """设置随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
    # 设置随机种子
    set_seed(42)

    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    total_steps = len(train_data) * config["epoch"]

    # 加载BERT配置和模型
    bert_config = BertConfig.from_pretrained(config["bert_path"], num_labels=config["class_num"])
    model = BertNER.from_pretrained(config["bert_path"], config=bert_config, model_config=config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器和学习率调度器
    optimizer = choose_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps
    )

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    best_f1 = 0.0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # BERT需要的输入：input_ids, attention_mask, token_type_ids, labels
            input_ids, attention_mask, token_type_ids, labels = batch_data
            loss = model(input_ids, attention_mask, token_type_ids, labels)

            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # 更新学习率
            train_loss.append(loss.item())

            if index % int(len(train_data) / 2) == 0:
                logger.info(f"batch {index}, loss {loss.item()}")

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        # 在验证集上评估
        micro_f1 = evaluator.eval(epoch)

        # 保存最好的模型
        if micro_f1 > best_f1:
            best_f1 = micro_f1
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"保存最佳模型，当前最佳Micro-F1: {best_f1}")

    logger.info(f"训练结束，最佳Micro-F1: {best_f1}")
    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)