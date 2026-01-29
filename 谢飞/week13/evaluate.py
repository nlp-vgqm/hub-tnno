# -*- coding: utf-8 -*-

"""
模型评估器
计算准确率、召回率和 F1 分数
"""

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data


class Evaluator:
    """模型效果测试"""
    
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
    
    def eval(self, epoch):
        """评估模型"""
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }
        
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() if isinstance(d, torch.Tensor) else tuple(t.cuda() for t in d) for d in batch_data]
            x, labels = batch_data  
            with torch.no_grad():
                pred_results = self.model(x)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return
    
    def write_stats(self, labels, pred_results, sentences):
        """统计预测结果"""
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return
    
    def show_stats(self):
        """显示统计结果"""
        F1_scores = []
        
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 准确率 = 正确识别数 / 识别出实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            # 召回率 = 正确识别数 / 样本实体数
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            # F1 分数
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            
            self.logger.info(
                "%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1)
            )
        
        # Macro-F1（宏平均）
        macro_f1 = np.mean(F1_scores)
        self.logger.info("Macro-F1: %f" % macro_f1)
        
        # Micro-F1（微平均）
        correct_pred = sum([
            self.stats_dict[key]["正确识别"]
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]
        ])
        total_pred = sum([
            self.stats_dict[key]["识别出实体数"]
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]
        ])
        true_enti = sum([
            self.stats_dict[key]["样本实体数"]
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]
        ])
        
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        
        self.logger.info("Micro-F1: %f" % micro_f1)
        self.logger.info("--------------------")
        return
    
    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        """解码标签序列为实体"""
        # 过滤掉 -1（忽略标签），只取有效标签
        valid_labels = [label for label in labels[:len(sentence)] if label != -1 and 0 <= label <= 8]
        # 如果长度不够，用 O (8) 填充
        if len(valid_labels) < len(sentence):
            valid_labels.extend([8] * (len(sentence) - len(valid_labels)))
        elif len(valid_labels) > len(sentence):
            valid_labels = valid_labels[:len(sentence)]
        
        labels = "".join([str(x) for x in valid_labels])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
