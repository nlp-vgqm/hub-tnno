# -*- coding: utf-8 -*-
"""
数据加载（LoRA + BERT NER 版本）

数据格式：每行 "字 标签"，句子间空行分隔
例如：
他 O
说 O
中 B-ORGANIZATION
国 I-ORGANIZATION

...

对齐策略：
- 数据本身是按“字”标注（word-level = char-level）
- 使用 HuggingFace fast tokenizer 的 word_ids() 做 subword 对齐
- 默认：只给每个字的第一个 subword 分配标签，后续 subword 置为 -100（忽略 loss）
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_schema(schema_path: str):
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    # schema: label(str) -> id(int)
    id2label = {v: k for k, v in schema.items()}
    return schema, id2label


def read_conll_like(path: str):
    """读取本项目 ner_data/{train,dev,test} 格式"""
    sentences = []
    labels = []
    with open(path, encoding="utf-8") as f:
        seg_chars = []
        seg_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if seg_chars:
                    sentences.append(seg_chars)
                    labels.append(seg_labels)
                    seg_chars, seg_labels = [], []
                continue
            ch, lab = line.split()
            seg_chars.append(ch)
            seg_labels.append(lab)
        if seg_chars:
            sentences.append(seg_chars)
            labels.append(seg_labels)
    return sentences, labels


class NERDataset(Dataset):
    def __init__(self, data_path: str, config: dict, tokenizer, label2id: dict):
        self.config = config
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.sentences, self.label_strs = read_conll_like(data_path)

        # 将标签字符串转成 id（按字）
        self.label_ids = []
        for labs in self.label_strs:
            self.label_ids.append([self.label2id[x] for x in labs])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars = self.sentences[idx]            # list[str]，每个元素是一个字
        word_labels = self.label_ids[idx]      # list[int]，长度与 chars 相同

        enc = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_attention_mask=True,
        )
        word_ids = enc.word_ids()

        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)   # special tokens
            else:
                if word_id != prev_word_id:
                    aligned_labels.append(word_labels[word_id])
                else:
                    # 同一个字被拆成多个 subword：后续 subword 忽略
                    aligned_labels.append(-100)
            prev_word_id = word_id

        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }
        return item


def build_tokenizer(config: dict):
    return AutoTokenizer.from_pretrained(config["pretrained_model"], use_fast=True)


def load_data(data_path: str, config: dict, shuffle=True):
    """返回 (dataloader, dataset, tokenizer, schema, id2label)"""
    schema, id2label = load_schema(config["schema_path"])
    tokenizer = build_tokenizer(config)
    dataset = NERDataset(data_path, config, tokenizer, schema)

    # 动态 padding（token classification 用这个 collate 更稳）
    def collate_fn(batch):
        # tokenizer.pad 支持 input_ids/attention_mask；labels 我们手动 pad -100
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            pad_len = max_len - x["input_ids"].shape[0]
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([x["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)]))
            labels.append(torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    dl = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    return dl, dataset, tokenizer, schema, id2label
