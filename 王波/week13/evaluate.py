# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 加载BERT tokenizer用于解码
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.label_mapping = {
            0: "B-PERSON", 1: "B-LOCATION", 2: "B-ORGANIZATION", 3: "B-TIME",
            4: "I-PERSON", 5: "I-LOCATION", 6: "I-ORGANIZATION", 7: "I-TIME",
            8: "O"
        }

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels, attention_mask = batch_data

            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask=attention_mask)

            self.write_stats(labels, pred_results, sentences, input_ids, attention_mask)

        self.show_stats()
        return

    def write_stats(self, true_labels, pred_results, sentences, input_ids, attention_mask):
        assert len(true_labels) == len(pred_results) == len(sentences)

        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)

        batch_size = len(sentences)
        for i in range(batch_size):
            sentence = sentences[i]
            true_label = true_labels[i].cpu().detach().tolist()

            # 获取预测标签
            if self.config["use_crf"]:
                pred_label = pred_results[i]
            else:
                pred_label = pred_results[i].cpu().detach().tolist()

            input_id = input_ids[i].cpu().detach().tolist()
            mask = attention_mask[i].cpu().detach().tolist()

            # 解码时过滤掉特殊标记
            true_entities = self.decode_from_tokens(sentence, input_id, true_label, mask)
            pred_entities = self.decode_from_tokens(sentence, input_id, pred_label, mask)

            # 统计指标
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def decode_from_tokens(self, sentence, token_ids, labels, mask):
        """
        从tokenized序列解码实体，正确处理CLS/SEP和subword
        """
        # 将token_ids转换为tokens
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # 过滤特殊标记和padding
        valid_tokens = []
        valid_labels = []

        for i, (token, label, mask_val) in enumerate(zip(tokens, labels, mask)):
            if mask_val == 0:  # padding
                continue
            if token in ["[CLS]", "[SEP]"]:  # 特殊标记
                continue

            # 处理子词：如果是##开头的子词，合并到前一个token
            if token.startswith("##"):
                if valid_tokens:
                    valid_tokens[-1] += token[2:]
            else:
                valid_tokens.append(token)
                valid_labels.append(label)

        # 现在valid_labels和valid_tokens已经对齐，可以解码实体
        return self.decode_entities(valid_tokens, valid_labels)

    def decode_entities(self, tokens, labels):
        """
        从对齐的tokens和labels解码实体
        """
        # 标签映射关系
        # 0:B-PER, 1:B-LOC, 2:B-ORG, 3:B-TIME
        # 4:I-PER, 5:I-LOC, 6:I-ORG, 7:I-TIME
        # 8:O

        results = defaultdict(list)

        i = 0
        while i < len(labels):
            label = labels[i]

            if label < 8:  # 实体标签
                entity_type = ""
                entity_tokens = []

                # 确定实体类型
                if label == 0 or label == 4:
                    entity_type = "PERSON"
                elif label == 1 or label == 5:
                    entity_type = "LOCATION"
                elif label == 2 or label == 6:
                    entity_type = "ORGANIZATION"
                elif label == 3 or label == 7:
                    entity_type = "TIME"

                # 开始收集实体tokens
                while i < len(labels):
                    current_label = labels[i]

                    # 检查是否属于同一实体类型
                    is_same_entity = (
                            (current_label == 0 and entity_type == "PERSON") or
                            (current_label == 1 and entity_type == "LOCATION") or
                            (current_label == 2 and entity_type == "ORGANIZATION") or
                            (current_label == 3 and entity_type == "TIME") or
                            (current_label == 4 and entity_type == "PERSON") or
                            (current_label == 5 and entity_type == "LOCATION") or
                            (current_label == 6 and entity_type == "ORGANIZATION") or
                            (current_label == 7 and entity_type == "TIME")
                    )

                    if is_same_entity:
                        entity_tokens.append(tokens[i])
                        i += 1
                    else:
                        break

                # 如果有实体tokens，添加到结果
                if entity_tokens and entity_type:
                    entity_text = "".join(entity_tokens)
                    results[entity_type].append(entity_text)
            else:
                i += 1  # O标签，跳过

        return results

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))

        if F1_scores:
            self.logger.info("Macro-F1: %f" % np.mean(F1_scores))

        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return