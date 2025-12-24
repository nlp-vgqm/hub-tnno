# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from loader import load_data
import os
import json

"""
模型效果评估
"""


# evaluate.py 修改部分
class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger

        # 直接使用原始训练数据构建知识库
        self.knwb_vectors, self.knwb_labels = self.build_knowledge_base_from_original_data()

        # 加载验证集
        if os.path.exists(config["valid_data_path"]):
            self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
            self.logger.info(f"验证集加载完成，共 {len(self.valid_data.dataset)} 个样本")
        else:
            self.logger.warning(f"验证集不存在: {config['valid_data_path']}")
            self.valid_data = None

    def build_knowledge_base_from_original_data(self):
        """直接从原始训练数据构建知识库"""
        original_data_path = self.config.get("original_train_data_path")
        if not original_data_path or not os.path.exists(original_data_path):
            self.logger.error(f"原始训练数据不存在: {original_data_path}")
            return None, None

        self.logger.info(f"从原始训练数据构建知识库: {original_data_path}")

        # 直接使用loader加载原始数据
        original_loader = load_data(original_data_path, self.config, shuffle=False)

        # 获取知识库数据
        knwb_data = original_loader.dataset

        if not hasattr(knwb_data, 'knwb'):
            self.logger.error("原始数据格式不正确，无法构建知识库")
            return None, None

        # 构建向量和标签
        all_vectors = []
        all_labels = []

        # 为每个标准问题构建向量
        for label_idx, question_ids in knwb_data.knwb.items():
            if not question_ids:
                continue

            # 堆叠所有问题ID
            question_matrix = torch.stack(question_ids, dim=0)

            # 批量编码
            with torch.no_grad():
                if torch.cuda.is_available():
                    question_matrix = question_matrix.cuda()

                vectors = self.model(question_matrix, mode="encode")
                vectors = torch.nn.functional.normalize(vectors, dim=-1)

                # 计算平均向量作为该类别的代表
                avg_vector = vectors.mean(dim=0, keepdim=True)

                all_vectors.append(avg_vector.cpu())
                all_labels.append(label_idx)

        if not all_vectors:
            return None, None

        knwb_vectors = torch.cat(all_vectors, dim=0)
        self.logger.info(f"知识库构建完成，共 {len(all_labels)} 个类别")

        return knwb_vectors, all_labels

    def eval(self, epoch):
        """评估函数保持不变"""
        if self.valid_data is None or self.knwb_vectors is None:
            self.logger.warning(f"epoch {epoch}: 验证集或知识库为空，跳过评估")
            return 0.0

        self.logger.info(f"开始第{epoch}轮模型评估")
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data):
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]

                input_ids, labels = batch_data
                batch_size = input_ids.shape[0]

                # 获取验证集数据的向量表示
                query_vectors = self.model(input_ids, mode="encode")
                query_vectors = torch.nn.functional.normalize(query_vectors, dim=-1)

                # 计算与知识库中所有向量的余弦相似度
                similarities = torch.mm(query_vectors, self.knwb_vectors.T)

                # 找到最相似的
                max_indices = torch.argmax(similarities, dim=1)

                # 获取预测的标签
                for i in range(batch_size):
                    max_idx = max_indices[i].item()

                    if max_idx < len(self.knwb_labels):
                        predicted_label = self.knwb_labels[max_idx]
                        true_label = labels[i].item()

                        if predicted_label == true_label:
                            correct += 1
                        total += 1

                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"评估进度: {batch_idx + 1}/{len(self.valid_data)} batches")

        if total > 0:
            accuracy = correct / total
            self.logger.info(f"验证集准确率: {accuracy:.4f} ({correct}/{total})")
        else:
            accuracy = 0.0
            self.logger.warning("验证集为空，无法计算准确率")

        self.model.train()
        return accuracy