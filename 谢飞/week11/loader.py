# -*- coding: utf-8 -*-

"""
数据加载器
用于SFT训练的数据加载
支持instruction-response格式
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader


class SFTDataset(Dataset):
    """SFT数据集"""
    
    def __init__(self, data_path, config, tokenizer):
        self.config = config
        self.path = data_path
        self.tokenizer = tokenizer
        self.data = []
        self.data_format = config.get("data_format", "instruction-response")
        self.load()
    
    def load(self):
        """加载数据"""
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    
                    if self.data_format == "instruction-response":
                        instruction = item.get("instruction", "")
                        response = item.get("response", "")
                    elif self.data_format == "input-output":
                        instruction = item.get("input", "")
                        response = item.get("output", "")
                    else:
                        # 兼容旧格式：content-title
                        instruction = item.get("content", "")
                        response = item.get("title", "")
                    
                    if instruction and response:
                        self.prepare_data(instruction, response)
                except json.JSONDecodeError:
                    continue
    
    def prepare_data(self, instruction, response):
        """准备数据"""
        # 构建输入文本：instruction + eos + response + eos
        text = f"{instruction}{self.tokenizer.eos_token}{response}{self.tokenizer.eos_token}"
        
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
        
        self.data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instruction": instruction,
            "response": response
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, tokenizer, shuffle=True):
    """
    用torch自带的DataLoader类封装数据
    Args:
        data_path: 数据文件路径
        config: 配置字典
        tokenizer: tokenizer对象
        shuffle: 是否打乱数据
    Returns:
        DataLoader 对象
    """
    dataset = SFTDataset(data_path, config, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "instructions": [item["instruction"] for item in batch],
            "responses": [item["response"] for item in batch]
        }
    )
    return dataloader


if __name__ == "__main__":
    from config import Config
    from model import SFTModel
    
    # 加载模型和tokenizer
    model = SFTModel(Config)
    tokenizer = model.tokenizer
    
    # 加载数据
    loader = load_data(Config["train_data_path"], Config, tokenizer)
    print(f"数据加载成功，共 {len(loader.dataset)} 条样本")
    
    # 测试一个batch
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("Input ids shape:", batch["input_ids"].shape)
        print("Labels shape:", batch["labels"].shape)
        print("Labels中非-100的数量:", (batch["labels"] != -100).sum().item())
        print("\n示例instruction:", batch["instructions"][0][:50])
        print("示例response:", batch["responses"][0][:50])
        break


