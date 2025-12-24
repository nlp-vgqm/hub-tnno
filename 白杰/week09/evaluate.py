# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from transformers import BertTokenizer

"""
模型效果测试 - 适配BERT
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()

        total_micro_f1 = 0.0
        batch_count = 0

        for index, batch_data in enumerate(self.valid_data):
            batch_count += 1
            # 获取原始句子
            start_idx = index * self.config["batch_size"]
            end_idx = start_idx + self.config["batch_size"]
            sentences = self.valid_data.dataset.sentences[start_idx:end_idx]

            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_ids, attention_mask, token_type_ids, labels = batch_data

            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask, token_type_ids)

            self.write_stats(labels, pred_results, sentences, attention_mask)

        # 计算并显示统计结果
        micro_f1 = self.show_stats()
        return micro_f1

    def write_stats(self, labels, pred_results, sentences, attention_mask):
        """处理标签和预测结果，计算各种指标"""
        batch_size = len(sentences)
        cuda_flag = torch.cuda.is_available()

        for i in range(batch_size):
            # 获取有效的标签和预测结果（排除padding部分）
            mask = attention_mask[i].cpu().numpy() == 1
            true_label = labels[i].cpu().numpy()[mask]
            pred_label = pred_results[i]

            # 确保长度一致
            if len(pred_label) > len(true_label):
                pred_label = pred_label[:len(true_label)]
            elif len(pred_label) < len(true_label):
                pred_label = pred_label + [0] * (len(true_label) - len(pred_label))

            # 解码实体
            true_entities = self.decode(sentences[i], true_label)
            pred_entities = self.decode(sentences[i], pred_label)

            # 更新统计信息
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 计算准确率、召回率和F1分数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))

        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))

        # 计算Micro-F1
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return micro_f1

    def decode(self, sentence, labels):
        """从标签序列解码出实体"""
        # 映射标签ID到标签名称
        id2schema = {v: k for k, v in self.valid_data.dataset.schema.items()}
        labels = [id2schema.get(int(l), "O") for l in labels[:len(sentence)]]

        results = defaultdict(list)
        current_entity = None
        current_type = None

        for i, (char, label) in enumerate(zip(sentence, labels)):
            if label.startswith("B-"):
                # 开始新实体
                if current_entity:
                    results[current_type].append(current_entity)
                current_type = label[2:]
                current_entity = char
            elif label.startswith("I-") and current_entity:
                # 实体继续
                current_entity += char
            else:
                # 实体结束或非实体
                if current_entity:
                    results[current_type].append(current_entity)
                    current_entity = None
                    current_type = None

        # 处理最后一个实体
        if current_entity:
            results[current_type].append(current_entity)

        return results