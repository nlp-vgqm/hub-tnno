# -*- coding: utf-8 -*-

"""
SFT (Supervised Fine-Tuning) 训练主程序
基于预训练生成模型进行监督微调
"""

import os
import numpy as np
import logging
import torch
from tqdm import tqdm

# 导入自定义模块
from config import Config
from model import SFTModel, choose_optimizer, get_learning_rate_scheduler
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
    
    # 设置随机种子
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
    
    # 加载模型
    logger.info("初始化SFT模型...")
    try:
        model = SFTModel(config)
        tokenizer = model.tokenizer
        logger.info("模型加载成功: %s" % config["model_name"])
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error("请检查config.py中的model_name配置")
        return
    
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
        device = torch.device("cuda")
    else:
        logger.info("使用CPU训练")
        device = torch.device("cpu")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型总参数数量: %d" % total_params)
    logger.info("可训练参数数量: %d" % trainable_params)
    
    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_data(config["train_data_path"], config, tokenizer, shuffle=True)
    logger.info("训练数据加载完成，共 %d 条样本" % len(train_data.dataset))
    
    # 加载验证数据
    logger.info("加载验证数据...")
    valid_data = load_data(config["valid_data_path"], config, tokenizer, shuffle=False)
    logger.info("验证数据加载完成，共 %d 条样本" % len(valid_data.dataset))
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 计算训练步数
    num_training_steps = len(train_data) * config["epoch"]
    if config.get("epoch_data_size"):
        num_training_steps = (config["epoch_data_size"] // config["batch_size"]) * config["epoch"]
    
    # 加载学习率调度器
    scheduler = get_learning_rate_scheduler(optimizer, config, num_training_steps)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger, valid_data)
    
    # 训练
    logger.info("开始训练...")
    logger.info("总训练步数: %d" % num_training_steps)
    
    global_step = 0
    epoch_data_size = config.get("epoch_data_size")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("=" * 60)
        logger.info("Epoch %d/%d 开始训练" % (epoch, config["epoch"]))
        logger.info("=" * 60)
        
        train_loss = []
        optimizer.zero_grad()
        
        max_batches = None
        if epoch_data_size is not None:
            max_batches = (epoch_data_size + config["batch_size"] - 1) // config["batch_size"]
            logger.info("本轮训练将处理 %d 个batch（约 %d 条样本）" % (max_batches, epoch_data_size))
        
        progress_bar = tqdm(enumerate(train_data), total=len(train_data) if max_batches is None else max_batches, desc=f"Epoch {epoch}")
        
        for index, batch_data in progress_bar:
            if max_batches is not None and index >= max_batches:
                break
            
            # 移动到GPU
            if cuda_flag:
                batch_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
            
            # 前向传播
            loss = model(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                labels=batch_data["labels"]
            )
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            train_loss.append(loss.item() * gradient_accumulation_steps)
            
            # 梯度累积步数达到后更新参数
            if (index + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
                
                # 定期保存模型
                if global_step % config.get("save_steps", 500) == 0:
                    model_path = os.path.join(config["model_path"], f"sft_model_step_{global_step}.pth")
                    torch.save(model.state_dict(), model_path)
                    logger.info("模型已保存到: %s" % model_path)
                
                # 定期评估
                if global_step % config.get("eval_steps", 200) == 0:
                    evaluator.eval(global_step)
        
        avg_loss = np.mean(train_loss)
        logger.info("Epoch %d 平均损失: %.4f" % (epoch, avg_loss))
        
        # 每个epoch结束后评估
        evaluator.eval(epoch)
        
        # 保存模型
        model_path = os.path.join(config["model_path"], "sft_model_epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
        logger.info("模型已保存到: %s" % model_path)
    
    logger.info("训练完成！")
    return


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SFT (Supervised Fine-Tuning) 训练")
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info("  模型保存路径: %s" % Config["model_path"])
    logger.info("  训练数据路径: %s" % Config["train_data_path"])
    logger.info("  验证数据路径: %s" % Config["valid_data_path"])
    logger.info("  最大长度: %d" % Config["max_length"])
    logger.info("  批次大小: %d" % Config["batch_size"])
    logger.info("  梯度累积步数: %d" % Config.get("gradient_accumulation_steps", 1))
    logger.info("  学习率: %f" % Config["learning_rate"])
    logger.info("  预训练模型: %s" % Config["model_name"])
    logger.info("=" * 60)
    
    main(Config)


