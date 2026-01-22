# -*- coding: utf-8 -*-

"""
基于 BERT + LoRA 的命名实体识别（NER）模型训练主程序

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法：
- 只训练少量参数（LoRA适配器），冻结原始BERT参数
- 大幅减少显存占用和训练时间
- 保持与全量微调相近的性能

使用方法：
1. 确保数据文件在正确位置（data目录下）
2. 确保 BERT 模型路径正确（config.py 中的 bert_path）
3. 安装依赖：pip install peft torchcrf
4. 运行：python main.py

配置参数：
- bert_path: BERT 模型路径，默认使用 hfl/chinese-bert-wwm-ext
- lora_r: LoRA的秩，默认8（越小参数越少）
- lora_alpha: LoRA的缩放参数，默认16（通常为r的2倍）
- lora_dropout: LoRA的dropout率，默认0.1
- use_crf: 是否使用 CRF 层，默认 True
- batch_size: 批次大小，默认 16
- learning_rate: 学习率，默认 2e-4（LoRA可以使用稍大的学习率）
- epoch: 训练轮数，默认 10
"""

import os
import numpy as np
import logging
import torch
import shutil

# 导入自定义模块
from config import Config
from model import BertNERModelWithLoRA, choose_optimizer
from loader import load_data
from evaluate import Evaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def copy_data_files_if_needed(config):
    """如果数据文件不存在，尝试从week09复制"""
    # 尝试从week09复制数据
    week09_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "week09")
    
    files_to_check = {
        "schema_path": ("data/schema.json", "data/schema.json"),
        "train_data_path": ("data/train", "data/train"),
        "valid_data_path": ("data/test", "data/test"),
    }
    
    copied = False
    for key, (source_rel, target_rel) in files_to_check.items():
        target_path = config[key]
        source_path = os.path.join(week09_base, source_rel)
        
        if not os.path.exists(target_path):
            if os.path.exists(source_path):
                try:
                    # 确保目标目录存在
                    target_dir = os.path.dirname(target_path)
                    if target_dir and not os.path.exists(target_dir):
                        os.makedirs(target_dir, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(source_path, target_path)
                    logger.info("已复制文件: %s -> %s" % (source_rel, target_rel))
                    copied = True
                except Exception as e:
                    logger.error("复制文件失败 %s: %s" % (target_rel, e))
                    return False
            else:
                logger.warning("源文件不存在: %s" % source_path)
    
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
        logger.warning("部分数据文件可能缺失，请检查路径配置")
    
    # 检查必要文件是否存在
    required_files = [
        config["schema_path"],
        config["train_data_path"],
        config["valid_data_path"],
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error("文件不存在: %s" % file_path)
            logger.error("请确保数据文件在当前目录下，或修改config.py中的路径配置")
            return
    
    # 检查 BERT 模型路径
    bert_path = config["bert_path"]
    is_local_path = (
        os.path.isabs(bert_path) or
        (len(bert_path) > 1 and bert_path[1] == ':') or
        bert_path.startswith('/')
    )
    
    if is_local_path:
        if not os.path.exists(bert_path):
            logger.error("BERT 模型路径不存在: %s" % bert_path)
            return
        else:
            logger.info("使用本地BERT模型: %s" % bert_path)
    else:
        logger.info("使用Hugging Face模型: %s" % bert_path)
    
    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_data(config["train_data_path"], config)
    logger.info("训练数据加载完成，共 %d 条样本" % len(train_data.dataset))
    
    # 加载模型
    logger.info("初始化 BERT + LoRA NER 模型...")
    model = BertNERModelWithLoRA(config)
    
    # 标识是否使用 GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU 可以使用，迁移模型至 GPU")
        model = model.cuda()
    else:
        logger.info("使用 CPU 训练")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型总参数数量: %d" % total_params)
    logger.info("可训练参数数量: %d" % trainable_params)
    logger.info("可训练参数比例: %.2f%%" % (trainable_params / total_params * 100))
    logger.info("LoRA大幅减少了可训练参数，节省显存和训练时间！")
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练
    logger.info("开始训练...")
    # 检查是否需要数据采样
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
        # 如果设置了数据采样，限制训练的 batch 数量
        max_batches = None
        if epoch_data_size is not None:
            max_batches = (epoch_data_size + config["batch_size"] - 1) // config["batch_size"]
            logger.info("本轮训练将处理 %d 个 batch（约 %d 条样本）" % (max_batches, epoch_data_size))
        
        for index, batch_data in enumerate(train_data):
            # 如果设置了数据采样限制，达到限制后停止
            if max_batches is not None and index >= max_batches:
                break
                
            optimizer.zero_grad()
            
            if cuda_flag:
                batch_data = [d.cuda() if isinstance(d, torch.Tensor) else tuple(t.cuda() for t in d) for d in batch_data]
            
            x, labels = batch_data  
            # 计算损失
            loss = model(x, labels)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            # 每 100 个 batch 打印一次
            if (index + 1) % 100 == 0:
                total_batches = max_batches if max_batches else len(train_data)
                logger.info(
                    "Batch %d/%d, Loss: %.4f" % (index + 1, total_batches, loss.item())
                )
        
        avg_loss = np.mean(train_loss)
        logger.info("Epoch %d 平均损失: %.4f" % (epoch, avg_loss))
        
        # 评估模型
        evaluator.eval(epoch)
        
        # 保存模型
        # 注意：使用LoRA时，可以只保存适配器权重，也可以保存完整模型
        model_path = os.path.join(config["model_path"], "lora_ner_epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
        logger.info("模型已保存到: %s" % model_path)
        
        # 也可以使用peft的save_pretrained方法只保存LoRA权重
        lora_path = os.path.join(config["model_path"], "lora_weights_epoch_%d" % epoch)
        model.bert.save_pretrained(lora_path)
        logger.info("LoRA权重已保存到: %s" % lora_path)
    
    logger.info("训练完成！")
    return


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("基于 BERT + LoRA 的命名实体识别（NER）模型训练")
    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info("  模型保存路径: %s" % Config["model_path"])
    logger.info("  训练数据路径: %s" % Config["train_data_path"])
    logger.info("  验证数据路径: %s" % Config["valid_data_path"])
    logger.info("  最大长度: %d" % Config["max_length"])
    logger.info("  隐藏层大小: %d" % Config["hidden_size"])
    logger.info("  批次大小: %d" % Config["batch_size"])
    logger.info("  学习率: %f" % Config["learning_rate"])
    logger.info("  使用 CRF: %s" % Config["use_crf"])
    logger.info("  标签类别数: %d" % Config["class_num"])
    logger.info("  BERT 模型路径: %s" % Config["bert_path"])
    logger.info("  LoRA 配置:")
    logger.info("    - LoRA rank (r): %d" % Config["lora_r"])
    logger.info("    - LoRA alpha: %d" % Config["lora_alpha"])
    logger.info("    - LoRA dropout: %f" % Config["lora_dropout"])
    logger.info("    - LoRA target modules: %s" % Config["lora_target_modules"])
    logger.info("=" * 60)
    
    main(Config)
