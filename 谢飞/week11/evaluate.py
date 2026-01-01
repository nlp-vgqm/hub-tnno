# -*- coding: utf-8 -*-

"""
模型评估和文本生成
"""

import torch
import logging
from tqdm import tqdm


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config, model, logger, valid_data):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = valid_data
        self.tokenizer = model.tokenizer
    
    def decode_text(self, token_ids):
        """将token ids解码为文本"""
        # 转换为列表（如果是tensor）
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        # 解码为文本
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text
    
    def eval(self, step_or_epoch):
        """评估模型"""
        self.logger.info("开始测试（Step/Epoch: %d）" % step_or_epoch)
        self.model.eval()
        
        total_loss = 0
        sample_count = 0
        max_samples = 5  # 每次评估显示5个样本
        
        with torch.no_grad():
            for index, batch_data in enumerate(self.valid_data):
                # 移动到GPU（如果可用）
                if torch.cuda.is_available():
                    batch_data = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                                 for k, v in batch_data.items()}
                
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                
                # 计算损失
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += loss.item()
                sample_count += input_ids.size(0)
                
                # 显示生成结果（只显示前几个样本）
                if index == 0:
                    # 为每个样本单独生成（因为需要从instruction开始生成）
                    for i in range(min(max_samples, input_ids.size(0))):
                        instruction = batch_data["instructions"][i]
                        
                        # 准备输入：只包含instruction部分
                        instruction_text = f"{instruction}{self.tokenizer.eos_token}"
                        instruction_encoded = self.tokenizer(
                            instruction_text,
                            max_length=self.config["max_length"] // 2,  # 为生成留出空间
                            padding=False,
                            truncation=True,
                            return_tensors="pt"
                        )
                        
                        inst_input_ids = instruction_encoded["input_ids"]
                        inst_attention_mask = instruction_encoded["attention_mask"]
                        
                        if torch.cuda.is_available():
                            inst_input_ids = inst_input_ids.cuda()
                            inst_attention_mask = inst_attention_mask.cuda()
                        
                        # 生成文本
                        generated_ids = self.model.generate(
                            input_ids=inst_input_ids,
                            attention_mask=inst_attention_mask,
                            max_length=self.config["max_length"],
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True
                        )
                        
                        # 解码生成结果（只取生成的部分，去掉instruction部分）
                        generated_text = self.decode_text(generated_ids[0][inst_input_ids.shape[1]:])
                        # 真实回复
                        true_response = batch_data["responses"][i]
                        
                        self.logger.info("-" * 60)
                        self.logger.info("样本 %d:" % (i + 1))
                        self.logger.info("指令: %s" % (instruction[:100] + "..." if len(instruction) > 100 else instruction))
                        self.logger.info("生成回复: %s" % (generated_text[:200] + "..." if len(generated_text) > 200 else generated_text))
                        self.logger.info("真实回复: %s" % (true_response[:100] + "..." if len(true_response) > 100 else true_response))
                        
                        if i >= max_samples - 1:
                            break
        
        avg_loss = total_loss / len(self.valid_data) if len(self.valid_data) > 0 else 0
        self.logger.info("验证平均损失: %.4f" % avg_loss)
        
        self.model.train()
        return avg_loss


if __name__ == "__main__":
    # 测试代码
    from config import Config
    from model import SFTModel
    from loader import load_data
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 加载模型和数据
    model = SFTModel(Config)
    valid_data = load_data(Config["valid_data_path"], Config, model.tokenizer, shuffle=False)
    
    evaluator = Evaluator(Config, model, logger, valid_data)
    evaluator.eval(1)

