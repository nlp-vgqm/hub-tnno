# -*- coding: utf-8 -*-

"""
SFT (Supervised Fine-Tuning) 模型
基于预训练生成模型（GPT/T5等）进行监督微调
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)


class SFTModel(nn.Module):
    """SFT模型：基于预训练生成模型进行监督微调"""
    
    def __init__(self, config):
        super(SFTModel, self).__init__()
        self.config = config
        model_name = config["model_name"]
        
        # 加载预训练模型和tokenizer
        try:
            # 尝试加载GPT-2模型
            if "gpt2" in model_name.lower():
                self.model = GPT2LMHeadModel.from_pretrained(model_name)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                # 设置pad_token（GPT-2默认没有pad_token）
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
            else:
                # 使用AutoModel加载其他模型
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            raise ValueError(f"无法加载模型 {model_name}: {e}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        Args:
            input_ids: 输入token ids (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            labels: 标签 (batch_size, seq_len)，用于计算损失
        Returns:
            如果提供了labels，返回损失值
            否则返回logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels is not None:
            return outputs.loss
        else:
            return outputs.logits
    
    def generate(self, input_ids, attention_mask=None, max_length=None, **kwargs):
        """
        生成文本
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            max_length: 最大生成长度
            **kwargs: 其他生成参数
        Returns:
            生成的token ids
        """
        if max_length is None:
            max_length = self.config["max_length"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                **kwargs
            )
        
        return outputs
    
    def prepare_inputs(self, instruction, response=None):
        """
        准备SFT训练输入
        SFT格式：instruction + response，使用特殊token分隔
        Args:
            instruction: 指令/输入文本
            response: 回复/输出文本（训练时提供，推理时不需要）
        Returns:
            input_ids, attention_mask, labels (如果response不为None)
        """
        # 构建输入文本
        if response is not None:
            # 训练模式：instruction + response
            # 格式：<instruction> <eos> <response> <eos>
            text = f"{instruction}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
        else:
            # 推理模式：只有instruction
            text = f"{instruction}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        if response is not None:
            # 创建labels：只对response部分计算损失
            # 找到instruction结束位置
            instruction_encoded = self.tokenizer(
                instruction,
                add_special_tokens=False,
                return_tensors="pt"
            )
            instruction_len = instruction_encoded["input_ids"].shape[1] + 1  # +1 for eos_token
            
            # 创建labels：instruction部分设为-100（忽略），response部分保留
            labels = input_ids.clone()
            labels[:instruction_len] = -100  # 忽略instruction部分
            
            # 将padding部分也设为-100
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return input_ids, attention_mask, labels
        else:
            return input_ids, attention_mask


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer_name = config["optimizer"]
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)
    
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_learning_rate_scheduler(optimizer, config, num_training_steps):
    """获取学习率调度器"""
    from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
    
    warmup_steps = config.get("warmup_steps", 0)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


if __name__ == "__main__":
    from config import Config
    model = SFTModel(Config)
    print("模型创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试准备输入
    instruction = "请介绍一下人工智能"
    response = "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
    
    input_ids, attention_mask, labels = model.prepare_inputs(instruction, response)
    print(f"\n输入形状: {input_ids.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签中非-100的数量: {(labels != -100).sum().item()}")

