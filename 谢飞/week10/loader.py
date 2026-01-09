# -*- coding: utf-8 -*-

"""
数据加载器
用于文本生成任务（content -> title）
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class TextGenerationDataset(Dataset):
    """文本生成数据集"""
    
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.data = []
        self.load()
    
    def load(self):
        """加载数据"""
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    content = item.get("content", "")
                    title = item.get("title", "")
                    
                    if content and title:
                        self.prepare_data(content, title)
                except json.JSONDecodeError:
                    continue
    
    def prepare_data(self, content, title):
        """准备数据：编码content和title"""
        # 编码content（Encoder输入）
        encoder_encoded = self.tokenizer(
            content,
            max_length=self.config["input_max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoder_input_ids = encoder_encoded["input_ids"].squeeze(0)
        encoder_attention_mask = encoder_encoded["attention_mask"].squeeze(0)
        
        # 编码title（Decoder输入和标签）
        # Decoder输入：[CLS] + title，用于teacher forcing
        # 注意：我们需要确保[CLS]在开头
        title_encoded = self.tokenizer(
            title,
            add_special_tokens=False,
            return_tensors="pt"
        )
        title_ids = title_encoded["input_ids"].squeeze(0).tolist()
        
        # Decoder输入：[CLS] + title，截断到max_length-1（为[SEP]留空间）
        decoder_input_ids = [self.tokenizer.cls_token_id] + title_ids[:self.config["output_max_length"]-2]
        decoder_input_ids = decoder_input_ids[:self.config["output_max_length"]-1]
        decoder_input_ids = decoder_input_ids + [self.tokenizer.pad_token_id] * (self.config["output_max_length"] - len(decoder_input_ids))
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        
        # Decoder attention mask
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id).long()
        
        # 标签：title + [SEP]，用于计算损失
        # 注意：标签需要shift，即预测下一个token
        labels = title_ids[:self.config["output_max_length"]-1] + [self.tokenizer.sep_token_id]
        labels = labels[:self.config["output_max_length"]]
        labels = labels + [self.tokenizer.pad_token_id] * (self.config["output_max_length"] - len(labels))
        labels = torch.LongTensor(labels)
        
        # 将padding位置的label设为-100（忽略计算损失）
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        
        self.data.append({
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path, config, shuffle=True):
    """
    用torch自带的DataLoader类封装数据
    Args:
        data_path: 数据文件路径
        config: 配置字典
        shuffle: 是否打乱数据
    Returns:
        DataLoader 对象
    """
    dataset = TextGenerationDataset(data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda batch: {
            "encoder_input_ids": torch.stack([item["encoder_input_ids"] for item in batch]),
            "encoder_attention_mask": torch.stack([item["encoder_attention_mask"] for item in batch]),
            "decoder_input_ids": torch.stack([item["decoder_input_ids"] for item in batch]),
            "decoder_attention_mask": torch.stack([item["decoder_attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }
    )
    return dataloader


if __name__ == "__main__":
    from config import Config
    loader = load_data(Config["train_data_path"], Config)
    print(f"数据加载成功，共 {len(loader.dataset)} 条样本")
    # 测试一个batch
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("Encoder input shape:", batch["encoder_input_ids"].shape)
        print("Decoder input shape:", batch["decoder_input_ids"].shape)
        print("Labels shape:", batch["labels"].shape)
        break

