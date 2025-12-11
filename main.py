# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 检查训练数据是否存在
    if not os.path.exists(config["train_data_path"]):
        logger.error(f"训练数据不存在: {config['train_data_path']}")
        logger.info("请先运行 create_triplets.py 生成三元组数据")
        return

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    logger.info(f"加载训练数据完成，共有 {len(train_data.dataset)} 个样本")

    # 加载模型
    model = SiameseNetwork(config)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    else:
        logger.info("GPU无法使用，使用CPU")

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    best_accuracy = 0.0  # 确保是float类型
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch {epoch} begin")
        train_loss = []

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # 三元组损失模式
            if config["use_triplet_loss"]:
                if len(batch_data) >= 3:
                    anchor, positive, negative = batch_data[:3]
                    loss = model(anchor, positive, negative, mode="triplet")
                else:
                    logger.warning(f"batch数据不足3个，跳过")
                    continue
            else:
                # 原有的二分类模式
                if len(batch_data) >= 3:
                    input_id1, input_id2, labels = batch_data[:3]
                    loss = model(input_id1, input_id2, labels, mode="pair")
                else:
                    logger.warning(f"batch数据不足3个，跳过")
                    continue

            train_loss.append(loss.item())

            # 每10个batch打印一次loss
            if index % 10 == 0:
                logger.info(f"batch {index}/{len(train_data)} loss: {loss.item():.4f}")

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        avg_loss = np.mean(train_loss)
        logger.info(f"epoch {epoch} average loss: {avg_loss:.4f}")

        # 每个epoch结束后评估
        accuracy = evaluator.eval(epoch)

        # 确保accuracy是数值类型
        if accuracy is None:
            logger.warning(f"epoch {epoch} 评估返回None，跳过模型保存")
            accuracy = 0.0

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"新的最佳模型已保存，准确率: {accuracy:.4f}")

        # 保存当前epoch模型
        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"epoch {epoch} 模型已保存")

    logger.info(f"训练完成，最佳准确率: {best_accuracy:.4f}")
    return


if __name__ == "__main__":
    main(Config)