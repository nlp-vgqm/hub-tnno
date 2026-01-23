# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
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

    # 加载训练数据
    logger.info(f"开始加载训练数据，路径: {config['train_data_path']}")
    train_data = load_data(config["train_data_path"], config)

    if len(train_data) == 0:
        logger.error("训练数据为空，请检查数据路径和格式")
        return None, None

    logger.info(f"训练数据加载完成，batch数量: {len(train_data)}")

    # 检查标签分布
    logger.info("检查训练数据标签分布...")
    all_labels = []
    for batch_data in train_data:
        if len(batch_data) == 3:
            _, labels, _ = batch_data
        else:
            _, labels = batch_data

        # 统计标签
        labels_flat = labels.flatten().cpu().numpy()
        all_labels.extend(labels_flat[labels_flat != -1])  # 排除padding

    if all_labels:
        from collections import Counter
        label_counts = Counter(all_labels)
        logger.info(f"标签分布: {label_counts}")

        # 检查是否有实体标签
        entity_labels = sum([count for label, count in label_counts.items() if label < 8])
        if entity_labels == 0:
            logger.warning("警告：训练数据中没有找到实体标签！")
        else:
            logger.info(f"实体标签总数: {entity_labels}")

    # 加载模型
    logger.info("开始加载模型...")
    model = TorchModel(config)

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"参数压缩比: {(trainable_params / total_params) * 100:.2f}%")

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    else:
        logger.info("使用CPU训练")

    # 加载优化器
    logger.info("创建优化器...")
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    logger.info("创建评估器...")
    evaluator = Evaluator(config, model, logger)

    # 训练
    logger.info(f"开始训练，共 {config['epoch']} 个epoch")
    best_f1 = 0
    best_epoch = 0

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("=" * 50)
        logger.info(f"Epoch {epoch}/{config['epoch']} 开始")
        train_loss = []

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # 现在batch_data包含三个元素：input_id, labels, attention_mask
            if len(batch_data) == 3:
                input_id, labels, attention_mask = batch_data
                loss = model(input_id, labels, attention_mask)
            else:
                input_id, labels = batch_data
                loss = model(input_id, labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # 每10个batch打印一次loss
            if index % 10 == 0:
                logger.info(f"Batch {index}/{len(train_data)} loss: {loss.item():.4f}")

        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch {epoch} 完成，平均loss: {avg_loss:.4f}")

        # 评估
        evaluator.eval(epoch)

    # 保存模型
    model_path = os.path.join(config["model_path"], f"epoch_{config['epoch']}.pth")

    # 如果使用LoRA，需要保存LoRA权重和基础模型的配置
    if config.get("use_lora", False):
        # 保存LoRA权重
        lora_path = os.path.join(config["model_path"], f"lora_weights")
        model.bert.save_pretrained(lora_path)

        # 保存分类头
        classifier_state = {
            "fc": model.fc.state_dict(),
            "classify": model.classify.state_dict(),
            "crf_layer": model.crf_layer.state_dict() if config["use_crf"] else None,
        }
        torch.save(classifier_state, model_path)
        logger.info(f"LoRA权重保存到: {lora_path}")
    else:
        # 原始保存方式
        torch.save(model.state_dict(), model_path)

    logger.info(f"模型保存到: {model_path}")
    logger.info("训练完成！")
    return model, train_data


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("程序开始运行...")
    logger.info(f"使用LoRA: {Config.get('use_lora', False)}")
    logger.info(f"LoRA配置: r={Config.get('lora_r', 8)}, alpha={Config.get('lora_alpha', 32)}")

    try:
        model, train_data = main(Config)
        logger.info("训练成功完成！")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        raise