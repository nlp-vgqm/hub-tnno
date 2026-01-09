# -*- coding: utf-8 -*-

"""
基于BERT的自回归语言模型训练主程序
用于文本生成任务（content -> title）
"""

import os
import numpy as np
import logging
import torch

# 导入自定义模块
from config import Config
from model import BertEncoderDecoder, choose_optimizer
from loader import load_data
from evaluate import Evaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(config):
    """主训练函数"""
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"], exist_ok=True)
        logger.info("创建模型保存目录: %s" % config["model_path"])
    
    # 检查必要文件是否存在
    required_files = [
        config["train_data_path"],
        config["valid_data_path"],
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error("文件不存在: %s" % file_path)
            logger.error("请确保数据文件在当前目录下，或修改config.py中的路径配置")
            return
    
    # 检查BERT模型路径
    bert_path = config["bert_path"]
    is_local_path = (
        os.path.isabs(bert_path) or
        (len(bert_path) > 1 and bert_path[1] == ':') or
        bert_path.startswith('/')
    )
    
    if is_local_path:
        if not os.path.exists(bert_path):
            logger.error("BERT模型路径不存在: %s" % bert_path)
            return
        else:
            logger.info("使用本地BERT模型: %s" % bert_path)
    else:
        logger.info("使用Hugging Face模型: %s" % bert_path)
    
    # 设置随机种子
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
    
    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_data(config["train_data_path"], config, shuffle=True)
    logger.info("训练数据加载完成，共 %d 条样本" % len(train_data.dataset))
    
    # 加载验证数据
    logger.info("加载验证数据...")
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)
    logger.info("验证数据加载完成，共 %d 条样本" % len(valid_data.dataset))
    
    # 加载模型
    logger.info("初始化BERT Encoder-Decoder模型...")
    model = BertEncoderDecoder(config)
    
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    else:
        logger.info("使用CPU训练")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型总参数数量: %d" % total_params)
    logger.info("可训练参数数量: %d" % trainable_params)
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, valid_data)
    
    # 训练
    logger.info("开始训练...")
    epoch_data_size = config.get("epoch_data_size")
    if epoch_data_size is not None:
        logger.info("每轮训练将只使用 %d 条样本（用于快速测试）" % epoch_data_size)
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("=" * 60)
        logger.info("Epoch %d/%d 开始训练" % (epoch, config["epoch"]))
        logger.info("=" * 60)
        
        train_loss = []
        max_batches = None
        if epoch_data_size is not None:
            max_batches = (epoch_data_size + config["batch_size"] - 1) // config["batch_size"]
            logger.info("本轮训练将处理 %d 个batch（约 %d 条样本）" % (max_batches, epoch_data_size))
        
        for index, batch_data in enumerate(train_data):
            if max_batches is not None and index >= max_batches:
                break
            
            optimizer.zero_grad()
            
            if cuda_flag:
                batch_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
            
            # 前向传播
            loss = model(
                encoder_input_ids=batch_data["encoder_input_ids"],
                encoder_attention_mask=batch_data["encoder_attention_mask"],
                decoder_input_ids=batch_data["decoder_input_ids"],
                decoder_attention_mask=batch_data["decoder_attention_mask"],
                labels=batch_data["labels"]
            )
            
            train_loss.append(loss.item())
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 每50个batch打印一次
            if (index + 1) % 50 == 0:
                total_batches = max_batches if max_batches else len(train_data)
                logger.info(
                    "Batch %d/%d, Loss: %.4f" % (index + 1, total_batches, loss.item())
                )
        
        avg_loss = np.mean(train_loss)
        logger.info("Epoch %d 平均损失: %.4f" % (epoch, avg_loss))
        
        # 评估模型
        evaluator.eval(epoch)
        
        # 保存模型
        model_path = os.path.join(config["model_path"], "bert_generation_epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
        logger.info("模型已保存到: %s" % model_path)
    
    logger.info("训练完成！")
    return


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("基于BERT的自回归语言模型训练（文本生成任务）")
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info("  模型保存路径: %s" % Config["model_path"])
    logger.info("  训练数据路径: %s" % Config["train_data_path"])
    logger.info("  验证数据路径: %s" % Config["valid_data_path"])
    logger.info("  输入最大长度: %d" % Config["input_max_length"])
    logger.info("  输出最大长度: %d" % Config["output_max_length"])
    logger.info("  隐藏层大小: %d" % Config["hidden_size"])
    logger.info("  Decoder层数: %d" % Config["num_decoder_layers"])
    logger.info("  批次大小: %d" % Config["batch_size"])
    logger.info("  学习率: %f" % Config["learning_rate"])
    logger.info("  BERT模型路径: %s" % Config["bert_path"])
    logger.info("=" * 60)
    
    main(Config)


