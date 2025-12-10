#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本匹配模型 - 使用三元组损失训练
功能模块：
1. 配置管理
2. 数据加载与预处理
3. 模型定义
4. 训练器
5. 评估器
6. 预测器
7. 主程序
"""

import json
import re
import os
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# ==================== 1. 配置管理模块 ====================

@dataclass
class ModelConfig:
    """模型配置类"""
    # 路径配置 
    model_path: str = "/Users/apple/Desktop/week08/model_output"
    train_data_path: str = "/Users/apple/Desktop/week08/train.json"
    valid_data_path: str = "/Users/apple/Desktop/week08/valid.json"
    vocab_path: str = "/Users/apple/Desktop/week08/chars.txt"

    # 模型参数
    max_length: int = 20
    hidden_size: int = 128
    embedding_dim: int = 128
    vocab_size: int = 5000
    dropout: float = 0.5
    encoder_type: str = "lstm"  # lstm, cnn, gru

    # 训练参数
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # adam, sgd
    weight_decay: float = 1e-4

    # 三元组参数
    margin: float = 0.2
    triplet_strategy: str = "random"  # random, semi-hard, hard
    epoch_data_size: int = 1000  # 每轮采样的三元组数量

    # 评估参数
    eval_batch_size: int = 64
    top_k: int = 1  # 检索时返回的top k结果

    def __post_init__(self):
        """后初始化处理"""
        os.makedirs(self.model_path, exist_ok=True)


# ==================== 2. 数据加载与预处理模块 ====================

class Vocabulary:
    """词汇表管理类"""

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

        if os.path.exists(vocab_path):
            self.load_vocab()

    def load_vocab(self):
        """加载词汇表"""
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def build_from_corpus(self, texts: List[str], max_vocab_size: int = 5000):
        """从语料构建词汇表"""
        word_counter = Counter()

        for text in texts:
            # 字符级分词
            for char in text:
                word_counter[char] += 1

        # 保留高频词
        most_common = word_counter.most_common(max_vocab_size - 2)

        for word, freq in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        # 保存词汇表
        self.save_vocab()
        return self.vocab_size

    def save_vocab(self):
        """保存词汇表"""
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for idx in range(2, self.vocab_size):
                f.write(self.idx2word[idx] + '\n')

    def encode(self, text: str, max_length: int) -> List[int]:
        """编码文本为索引序列"""
        # 简单清洗
        text = re.sub(r'\s+', ' ', text.strip())

        # 字符级编码
        indices = []
        for char in text:
            indices.append(self.word2idx.get(char, self.word2idx['<UNK>']))

        # 截断或填充
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (max_length - len(indices))

        return indices

    def __len__(self):
        return self.vocab_size


class TripletDataset(Dataset):
    """三元组数据集"""

    def __init__(self, config: ModelConfig, mode: str = 'train'):
        """
        初始化数据集

        Args:
            config: 模型配置
            mode: 'train' 或 'valid'
        """
        self.config = config
        self.mode = mode
        self.vocab = Vocabulary(config.vocab_path)

        # 加载数据
        if mode == 'train':
            data_path = config.train_data_path
        else:
            data_path = config.valid_data_path

        self.load_data(data_path)

        # 构建知识库（标准问题到问题列表的映射）
        self.build_knowledge_base()

    def load_data(self, data_path: str):
        """加载数据"""
        self.data = []
        self.labels = []

        if not os.path.exists(data_path):
            print(f"警告：数据文件 {data_path} 不存在")
            return

        # 根据文件内容格式加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            try:
                # 尝试解析为JSON
                items = json.loads(content)

                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            # 训练数据格式：{"target": "类别", "questions": ["问题1", "问题2"]}
                            label = item.get("target", "")
                            questions = item.get("questions", [])

                            for question in questions:
                                if question and label:
                                    encoded = self.vocab.encode(question, self.config.max_length)
                                    self.data.append(encoded)
                                    self.labels.append(label)
                        elif isinstance(item, list) and len(item) >= 2:
                            # 验证数据格式：["问题", "类别"]
                            question, label = item[0], item[1]
                            if question and label:
                                encoded = self.vocab.encode(question, self.config.max_length)
                                self.data.append(encoded)
                                self.labels.append(label)
                elif isinstance(items, dict):
                    # 如果是单个字典对象
                    label = items.get("target", "")
                    questions = items.get("questions", [])
                    for question in questions:
                        if question and label:
                            encoded = self.vocab.encode(question, self.config.max_length)
                            self.data.append(encoded)
                            self.labels.append(label)

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
            except Exception as e:
                print(f"解析数据失败: {e}")

    def build_knowledge_base(self):
        """构建知识库：标签到问题索引的映射"""
        self.label_to_indices = defaultdict(list)
        self.label_list = []

        for idx, label in enumerate(self.labels):
            if label not in self.label_list:
                self.label_list.append(label)
            label_idx = self.label_list.index(label)
            self.label_to_indices[label_idx].append(idx)

    def get_triplet(self, label_idx: int) -> Tuple[List[int], List[int], List[int]]:
        """生成一个三元组"""
        # 获取当前类别的所有样本
        pos_indices = self.label_to_indices[label_idx]

        # 随机选择锚点和正样本
        anchor_idx, positive_idx = random.sample(pos_indices, 2)
        anchor = self.data[anchor_idx]
        positive = self.data[positive_idx]

        # 选择负样本
        # 随机选择其他类别
        other_labels = [l for l in range(len(self.label_list)) if l != label_idx]
        if not other_labels:
            # 如果没有其他类别，使用同一个类别（这种情况应该避免）
            negative_idx = random.choice(pos_indices)
        else:
            negative_label = random.choice(other_labels)
            negative_idx = random.choice(self.label_to_indices[negative_label])

        negative = self.data[negative_idx]

        return anchor, positive, negative

    def __len__(self):
        if self.mode == 'train':
            return self.config.epoch_data_size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 训练时随机生成三元组
            label_idx = random.randint(0, len(self.label_list) - 1)
            anchor, positive, negative = self.get_triplet(label_idx)

            return (
                torch.LongTensor(anchor),
                torch.LongTensor(positive),
                torch.LongTensor(negative)
            )
        else:
            # 验证时返回单个样本和标签
            return (
                torch.LongTensor(self.data[idx]),
                torch.LongTensor([self.label_list.index(self.labels[idx])])
            )


class DataModule:
    """数据管理模块"""

    def __init__(self, config: ModelConfig):
        self.config = config

        # 构建词汇表（如果不存在）
        if not os.path.exists(config.vocab_path):
            print("构建词汇表...")
            # 需要先加载数据来构建词汇表
            self.prepare_vocab()

        # 初始化数据集
        self.train_dataset = TripletDataset(config, mode='train')
        self.valid_dataset = TripletDataset(config, mode='valid')

        # 更新配置中的词汇表大小
        config.vocab_size = len(self.train_dataset.vocab)

    def prepare_vocab(self):
        """准备词汇表"""
        # 收集所有文本
        all_texts = []

        for data_path in [self.config.train_data_path, self.config.valid_data_path]:
            if os.path.exists(data_path):
                try:
                    with open(data_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        items = json.loads(content)

                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    all_texts.extend(item.get("questions", []))
                                elif isinstance(item, list) and len(item) >= 2:
                                    all_texts.append(item[0])
                        elif isinstance(items, dict):
                            all_texts.extend(items.get("questions", []))
                except Exception as e:
                    print(f"加载数据文件 {data_path} 失败: {e}")

        # 构建词汇表
        vocab = Vocabulary(self.config.vocab_path)
        vocab.build_from_corpus(all_texts, max_vocab_size=5000)

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """获取数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Mac上设置为0避免问题
        )

        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, valid_loader


# ==================== 3. 模型定义模块 ====================

class SentenceEncoder(nn.Module):
    """句子编码器"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=0
        )

        # 选择编码器类型
        if config.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                config.embedding_dim,
                config.hidden_size,
                batch_first=True,
                bidirectional=True
            )
            output_dim = config.hidden_size * 2

        elif config.encoder_type == "gru":
            self.encoder = nn.GRU(
                config.embedding_dim,
                config.hidden_size,
                batch_first=True,
                bidirectional=True
            )
            output_dim = config.hidden_size * 2

        elif config.encoder_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(config.embedding_dim, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(128, config.hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            output_dim = config.hidden_size

        else:
            raise ValueError(f"不支持的编码器类型: {config.encoder_type}")

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len)

        Returns:
            编码后的向量，形状为 (batch_size, hidden_size)
        """
        # 嵌入层
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        if self.config.encoder_type in ["lstm", "gru"]:
            # RNN编码器
            encoded, _ = self.encoder(embedded)

            # 使用最大池化获取句子表示
            pooled = F.max_pool1d(
                encoded.transpose(1, 2),  # (batch, hidden*2, seq_len)
                kernel_size=encoded.size(1)
            ).squeeze(2)  # (batch, hidden*2)

        else:  # CNN
            # 转置以适应CNN输入格式
            embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)

            # CNN编码器
            encoded = self.encoder(embedded)
            pooled = encoded.squeeze(2)  # (batch, hidden_size)

        # 输出层
        output = self.output_layer(pooled)

        return output


class TripletLoss(nn.Module):
    """三元组损失"""

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self,
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        计算三元组损失

        Args:
            anchor: 锚点向量 (batch_size, hidden_dim)
            positive: 正样本向量 (batch_size, hidden_dim)
            negative: 负样本向量 (batch_size, hidden_dim)

        Returns:
            损失值
        """
        # 计算距离
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)

        # 三元组损失
        losses = F.relu(pos_distance - neg_distance + self.margin)

        return losses.mean()


class SiameseNetwork(nn.Module):
    """孪生网络（用于三元组训练）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 句子编码器
        self.encoder = SentenceEncoder(config)

        # 损失函数
        self.loss_fn = TripletLoss(margin=config.margin)

    def forward(self,
                anchor: Optional[torch.Tensor] = None,
                positive: Optional[torch.Tensor] = None,
                negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        两种模式：
        1. 训练模式：传入anchor, positive, negative，返回损失
        2. 编码模式：只传入anchor，返回编码向量

        Args:
            anchor: 锚点输入
            positive: 正样本输入
            negative: 负样本输入

        Returns:
            如果是训练模式：损失值
            如果是编码模式：编码向量
        """
        if positive is not None and negative is not None:
            # 训练模式
            anchor_vec = self.encoder(anchor)
            positive_vec = self.encoder(positive)
            negative_vec = self.encoder(negative)

            return self.loss_fn(anchor_vec, positive_vec, negative_vec)

        elif anchor is not None:
            # 编码模式
            return self.encoder(anchor)

        else:
            raise ValueError("必须提供输入参数")

    def encode(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """编码文本"""
        return self.encoder(text_tensor)


# ==================== 4. 训练器模块 ====================

class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: ModelConfig, model: SiameseNetwork):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # 训练历史
        self.train_losses = []
        self.valid_accuracies = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="训练")
        for batch in pbar:
            # 移动到设备
            anchor, positive, negative = [x.to(self.device) for x in batch]

            # 前向传播
            loss = self.model(anchor, positive, negative)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches if num_batches > 0 else 0

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              evaluator: 'Evaluator',
              save_best: bool = True) -> Dict[str, List[float]]:
        """完整训练流程"""

        best_accuracy = 0

        for epoch in range(self.config.epochs):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"{'=' * 60}")

            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"训练损失: {train_loss:.4f}")

            # 评估
            accuracy = evaluator.evaluate(epoch + 1)
            self.valid_accuracies.append(accuracy)

            # 学习率调度
            self.scheduler.step(train_loss)

            # 保存最佳模型
            if save_best and accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(epoch + 1, accuracy, is_best=True)
                print(f"✓ 保存最佳模型，准确率: {accuracy:.4f}")

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, accuracy)

        # 保存最终模型
        self.save_checkpoint(self.config.epochs, accuracy, is_final=True)

        return {
            'train_losses': self.train_losses,
            'valid_accuracies': self.valid_accuracies
        }

    def save_checkpoint(self,
                        epoch: int,
                        accuracy: float,
                        is_best: bool = False,
                        is_final: bool = False):
        """保存检查点"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'valid_accuracies': self.valid_accuracies,
            'accuracy': accuracy,
            'config': self.config.__dict__
        }

        if is_best:
            filename = f"best_model_epoch{epoch}_acc{accuracy:.4f}.pth"
        elif is_final:
            filename = f"final_model_epoch{epoch}.pth"
        else:
            filename = f"checkpoint_epoch{epoch}.pth"

        save_path = os.path.join(self.config.model_path, filename)
        torch.save(checkpoint, save_path)
        print(f"✓ 模型保存到: {save_path}")


# ==================== 5. 评估器模块 ====================

class KnowledgeBase:
    """知识库管理"""

    def __init__(self, train_dataset: TripletDataset):
        self.train_dataset = train_dataset
        self.question_vectors = None
        self.label_vectors = None
        self.is_updated = False

    def build(self, model: SiameseNetwork, device: torch.device):
        """构建知识库向量"""
        model.eval()

        all_questions = []
        question_to_label = []

        # 收集所有训练问题和标签
        for idx in range(len(self.train_dataset.data)):
            question = self.train_dataset.data[idx]
            label = self.train_dataset.labels[idx]
            label_idx = self.train_dataset.label_list.index(label)

            all_questions.append(torch.LongTensor(question))
            question_to_label.append(label_idx)

        # 批量编码
        batch_size = 64
        all_vectors = []

        with torch.no_grad():
            for i in range(0, len(all_questions), batch_size):
                batch = torch.stack(all_questions[i:i + batch_size]).to(device)
                vectors = model.encode(batch)
                all_vectors.append(vectors.cpu())

        self.question_vectors = torch.cat(all_vectors, dim=0)
        self.question_labels = torch.LongTensor(question_to_label)
        self.is_updated = True

    def search(self,
               query_vector: torch.Tensor,
               top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """在知识库中搜索"""
        if not self.is_updated:
            raise ValueError("知识库未构建，请先调用build()方法")

        # 计算相似度（余弦相似度）
        query_vector = F.normalize(query_vector, dim=-1)
        kb_vectors = F.normalize(self.question_vectors, dim=-1)

        similarities = torch.matmul(query_vector, kb_vectors.T)

        # 获取top k结果
        top_scores, top_indices = torch.topk(similarities, k=top_k, dim=-1)

        # 获取对应的标签
        top_labels = self.question_labels[top_indices]

        return top_scores, top_labels


class Evaluator:
    """模型评估器"""

    def __init__(self,
                 config: ModelConfig,
                 model: SiameseNetwork,
                 train_dataset: TripletDataset,
                 valid_loader: DataLoader):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 知识库
        self.knowledge_base = KnowledgeBase(train_dataset)

        # 验证数据
        self.valid_loader = valid_loader

        # 评估结果
        self.results = []

    def evaluate(self, epoch: int) -> float:
        """评估模型"""
        print(f"评估第 {epoch} 轮模型...")

        # 更新知识库
        print("更新知识库向量...")
        self.knowledge_base.build(self.model, self.device)

        # 评估
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="评估"):
                texts, labels = batch
                texts = texts.to(self.device)
                labels = labels.to(self.device).squeeze()

                # 编码查询文本
                query_vectors = self.model.encode(texts)

                # 在知识库中搜索
                _, predicted_labels = self.knowledge_base.search(query_vectors, top_k=self.config.top_k)

                # 检查预测结果
                predicted_labels = predicted_labels.squeeze().to(self.device)

                # 如果是top_k > 1，检查是否包含正确答案
                if self.config.top_k > 1:
                    for i in range(len(labels)):
                        if labels[i] in predicted_labels[i]:
                            correct += 1
                else:
                    correct += (predicted_labels == labels).sum().item()

                total += len(labels)

        accuracy = correct / total if total > 0 else 0

        print(f"评估结果: {correct}/{total} = {accuracy:.4f}")

        # 记录结果
        self.results.append({
            'epoch': epoch,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })

        return accuracy

    def print_summary(self):
        """打印评估摘要"""
        if not self.results:
            print("暂无评估结果")
            return

        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)

        for result in self.results:
            print(f"Epoch {result['epoch']}: "
                  f"准确率 = {result['accuracy']:.4f} "
                  f"({result['correct']}/{result['total']})")

        best_result = max(self.results, key=lambda x: x['accuracy'])
        print(f"\n最佳模型: Epoch {best_result['epoch']}, "
              f"准确率 = {best_result['accuracy']:.4f}")


# ==================== 6. 预测器模块 ====================

class Predictor:
    """预测器（用于实际应用）"""

    def __init__(self,
                 config: ModelConfig,
                 model: SiameseNetwork,
                 knowledge_base: KnowledgeBase):
        self.config = config
        self.model = model
        self.knowledge_base = knowledge_base
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.idx_to_label = {idx: label for idx, label in enumerate(knowledge_base.train_dataset.label_list)}

    def predict(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """预测文本类别"""
        # 编码文本
        vocab = self.knowledge_base.train_dataset.vocab
        encoded = vocab.encode(text, self.config.max_length)
        text_tensor = torch.LongTensor([encoded]).to(self.device)

        with torch.no_grad():
            # 获取文本向量
            text_vector = self.model.encode(text_tensor)

            # 在知识库中搜索
            scores, label_indices = self.knowledge_base.search(text_vector, top_k=top_k)

        # 准备结果
        results = []
        for i in range(min(top_k, len(label_indices[0]))):
            label_idx = label_indices[0][i].item()
            score = scores[0][i].item()
            label = self.idx_to_label.get(label_idx, f"未知标签_{label_idx}")

            results.append({
                'label': label,
                'score': score,
                'confidence': self._score_to_confidence(score)
            })

        return results

    def _score_to_confidence(self, score: float) -> str:
        """将分数转换为置信度描述"""
        if score > 0.9:
            return "非常高"
        elif score > 0.7:
            return "高"
        elif score > 0.5:
            return "中等"
        elif score > 0.3:
            return "低"
        else:
            return "非常低"

    def interactive_mode(self):
        """交互式预测模式"""
        print("\n" + "=" * 60)
        print("交互式预测模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        while True:
            try:
                text = input("\n请输入要分类的文本: ").strip()

                if text.lower() in ['quit', 'exit', 'q']:
                    print("退出预测模式")
                    break

                if not text:
                    print("输入不能为空")
                    continue

                # 预测
                results = self.predict(text, top_k=3)

                print(f"\n预测结果:")
                for i, result in enumerate(results, 1):
                    print(f"{i}. 类别: {result['label']}, "
                          f"相似度: {result['score']:.4f}, "
                          f"置信度: {result['confidence']}")

            except KeyboardInterrupt:
                print("\n退出预测模式")
                break
            except Exception as e:
                print(f"预测出错: {e}")


# ==================== 7. 主程序模块 ====================

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/Users/apple/Desktop/week08/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """主函数"""
    logger = setup_logging()

    print("=" * 80)
    print("文本匹配模型 - 使用三元组损失训练")
    print("=" * 80)

    try:
        # 1. 加载配置
        print("\n1. 加载配置...")
        config = ModelConfig()
        logger.info(f"配置加载完成: {config}")

        # 2. 准备数据
        print("\n2. 准备数据...")
        data_module = DataModule(config)
        train_loader, valid_loader = data_module.get_dataloaders()

        train_dataset = data_module.train_dataset
        valid_dataset = data_module.valid_dataset

        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(valid_dataset)}")
        logger.info(f"类别数量: {len(train_dataset.label_list)}")

        # 3. 初始化模型
        print("\n3. 初始化模型...")
        model = SiameseNetwork(config)
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 4. 初始化训练器
        print("\n4. 初始化训练器...")
        trainer = ModelTrainer(config, model)

        # 5. 初始化评估器
        print("\n5. 初始化评估器...")
        evaluator = Evaluator(config, model, train_dataset, valid_loader)

        # 6. 训练模型
        print("\n6. 开始训练...")
        history = trainer.train(train_loader, valid_loader, evaluator, save_best=True)

        # 7. 打印评估结果
        print("\n7. 训练完成，评估结果:")
        evaluator.print_summary()

        # 8. 初始化预测器
        print("\n8. 初始化预测器...")
        predictor = Predictor(config, model, evaluator.knowledge_base)

        # 9. 交互式预测
        print("\n9. 启动交互式预测模式")
        predictor.interactive_mode()

        logger.info("程序运行完成")

    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        print(f"错误: {e}")


def quick_test():
    """快速测试（无需训练数据）"""
    print("快速测试模式...")

    # 创建简单配置
    config = ModelConfig(
        max_length=10,
        hidden_size=64,
        embedding_dim=64,
        vocab_size=100,
        epochs=1,
        batch_size=4,
        epoch_data_size=10
    )

    # 创建模拟数据
    vocab = Vocabulary("/Users/apple/Desktop/week08/test_vocab.txt")

    # 手动构建小型词汇表
    for i in range(100):
        vocab.word2idx[f"char_{i}"] = i + 2
        vocab.idx2word[i + 2] = f"char_{i}"
    vocab.vocab_size = 102

    # 创建简单模型
    model = SiameseNetwork(config)

    # 创建模拟输入
    batch_size = 4
    seq_len = config.max_length

    anchor = torch.randint(2, 100, (batch_size, seq_len))
    positive = torch.randint(2, 100, (batch_size, seq_len))
    negative = torch.randint(2, 100, (batch_size, seq_len))

    print(f"输入形状: anchor={anchor.shape}")

    # 测试前向传播
    loss = model(anchor, positive, negative)
    print(f"三元组损失: {loss.item():.4f}")

    # 测试编码
    encoded = model.encode(anchor)
    print(f"编码输出形状: {encoded.shape}")

    print("✓ 快速测试通过")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文本匹配模型训练")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "predict"],
                        help="运行模式: train(训练), test(测试), predict(预测)")
    parser.add_argument("--model_path", type=str,
                        help="模型路径（预测模式时使用）")
    parser.add_argument("--text", type=str,
                        help="要预测的文本（预测模式时使用）")

    args = parser.parse_args()

    if args.mode == "test":
        quick_test()
    elif args.mode == "predict":
        # 预测模式
        if not args.model_path:
            print("错误: 预测模式需要指定 --model_path")
        elif not args.text:
            print("错误: 预测模式需要指定 --text")
        else:
            # 这里可以实现加载模型并进行预测的逻辑
            print(f"预测文本: {args.text}")
            print("注意: 完整预测功能需要训练好的模型")
    else:
        # 训练模式
        main()