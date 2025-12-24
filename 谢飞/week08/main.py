# -*- coding: utf-8 -*-

"""
使用三元组损失完成文本匹配模型训练
基于Triplet Loss的句子编码器模型

三元组损失（Triplet Loss）原理：
- 给定一个anchor（锚点）、一个positive（正样本，与anchor同类）、一个negative（负样本，与anchor不同类）
- 目标：让anchor和positive之间的距离小于anchor和negative之间的距离
- 损失函数：L = max(0, d(a,p) - d(a,n) + margin)
- 其中d是距离函数（欧氏距离或余弦距离），margin是正负样本之间的最小间隔

使用方法：
1. 确保数据文件在正确位置（data目录下）
2. 运行：python main.py

配置参数：
- margin: 三元组损失的margin参数，默认0.5
- distance_type: 距离类型，'euclidean'（欧氏距离）或'cosine'（余弦距离）
- hidden_size: 隐藏层大小，默认128
- batch_size: 批次大小，默认32
- learning_rate: 学习率，默认1e-3
- epoch: 训练轮数，默认10
"""

import os
import numpy as np
import logging
import torch

# 导入自定义模块
from config import Config
from model import TripletNetwork
from data_loader import load_triplet_data
from evaluate import Evaluator
from utils import choose_optimizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def copy_data_files_if_needed(config):
    """如果数据文件不存在，从原始位置复制"""
    import shutil
    
    # 原始数据路径
    original_base = r"E:\AI\lessons\第八周 文本匹配\week8 文本匹配问题\week8 文本匹配问题"
    
    # 需要检查的文件
    files_to_check = {
        "schema_path": ("data/schema.json", "data/schema.json"),
        "train_data_path": ("data/train.json", "data/train.json"),
        "valid_data_path": ("data/valid.json", "data/valid.json"),
        "vocab_path": ("chars.txt", "chars.txt"),
    }
    
    copied = False
    for key, (source_rel, target_rel) in files_to_check.items():
        target_path = config[key]
        source_path = os.path.join(original_base, source_rel)
        
        if not os.path.exists(target_path):
            if os.path.exists(source_path):
                try:
                    # 确保目标目录存在
                    target_dir = os.path.dirname(target_path)
                    if target_dir and not os.path.exists(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(source_path, target_path)
                    logger.info("已复制文件: %s" % target_rel)
                    copied = True
                except Exception as e:
                    logger.error("复制文件失败 %s: %s" % (target_rel, e))
                    return False
            else:
                logger.error("源文件不存在: %s" % source_path)
                return False
    
    if copied:
        logger.info("数据文件复制完成！")
    return True


def main(config):
    """主训练函数"""
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"], exist_ok=True)
        logger.info("创建模型保存目录: %s" % config["model_path"])
    
    # 检查并复制数据文件（如果需要）
    if not copy_data_files_if_needed(config):
        logger.error("数据文件准备失败，请检查路径配置")
        return
    
    # 再次检查数据文件是否存在
    required_files = [
        config["schema_path"],
        config["train_data_path"],
        config["valid_data_path"],
        config["vocab_path"]
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error("文件不存在: %s" % file_path)
            logger.error("请确保数据文件在当前目录下，或修改config.py中的路径配置")
            return
    
    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_triplet_data(config["train_data_path"], config)
    
    # 加载模型
    logger.info("初始化模型...")
    model = TripletNetwork(config)
    
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    else:
        logger.info("使用CPU训练")
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练
    logger.info("开始训练...")
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("=" * 60)
        logger.info("Epoch %d/%d 开始训练" % (epoch, config["epoch"]))
        logger.info("=" * 60)
        
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            anchor, positive, negative = batch_data
            
            # 计算三元组损失
            loss = model(anchor, positive, negative)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            # 每100个batch打印一次
            if (index + 1) % 100 == 0:
                logger.info("Batch %d/%d, Loss: %.4f" % (index + 1, len(train_data), loss.item()))
        
        avg_loss = np.mean(train_loss)
        logger.info("Epoch %d 平均损失: %.4f" % (epoch, avg_loss))
        
        # 评估模型
        evaluator.eval(epoch)
        
        # 保存模型
        model_path = os.path.join(config["model_path"], "triplet_epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
        logger.info("模型已保存到: %s" % model_path)
    
    logger.info("训练完成！")
    return


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("三元组损失文本匹配模型训练")
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info("  模型保存路径: %s" % Config["model_path"])
    logger.info("  训练数据路径: %s" % Config["train_data_path"])
    logger.info("  验证数据路径: %s" % Config["valid_data_path"])
    logger.info("  词汇表路径: %s" % Config["vocab_path"])
    logger.info("  最大长度: %d" % Config["max_length"])
    logger.info("  隐藏层大小: %d" % Config["hidden_size"])
    logger.info("  批次大小: %d" % Config["batch_size"])
    logger.info("  学习率: %f" % Config["learning_rate"])
    logger.info("  Margin: %f" % Config["margin"])
    logger.info("  距离类型: %s" % Config["distance_type"])
    logger.info("=" * 60)
    
    main(Config)
