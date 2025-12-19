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
            pred_label = pred_results[i] if not self.config["use_crf"] else pred_results[i]

            # 如果pred_label是torch.Tensor，转换为list
            if torch.is_tensor(pred_label):
                pred_label = pred_label.cpu().detach().tolist()

            input_id = input_ids[i].cpu().detach().tolist()
            mask = attention_mask[i].cpu().detach().tolist()

            # 解码时过滤掉特殊标记
            true_entities = self.decode_from_tokens(sentence, input_id, true_label, mask)
            pred_entities = self.decode_from_tokens(sentence, input_id, pred_label, mask)

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
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
                # 子词的标签需要保留，用于后续解码
                if valid_labels:
                    valid_labels.append(label)
                else:
                    valid_labels.append(8)  # O标签
            else:
                valid_tokens.append(token)
                valid_labels.append(label)

        # 现在valid_labels和valid_tokens已经对齐，可以解码实体
        return self.decode_entities(valid_tokens, valid_labels)

    def decode_entities(self, tokens, labels):
        """
        从对齐的tokens和labels解码实体
        """
        # 将标签序列转换为字符串用于正则匹配
        label_str = "".join([str(x) for x in labels])

        results = defaultdict(list)

        # 使用正则匹配找到连续的实体标签
        entity_patterns = {
            "LOCATION": (r"(04+)", r"0[4]*"),
            "ORGANIZATION": (r"(15+)", r"1[5]*"),
            "PERSON": (r"(26+)", r"2[6]*"),
            "TIME": (r"(37+)", r"3[7]*")
        }

        for entity_type, (pattern_str, _) in entity_patterns.items():
            pattern = re.compile(pattern_str)
            for match in pattern.finditer(label_str):
                start, end = match.span()
                entity_tokens = tokens[start:end]
                entity_text = "".join(entity_tokens)
                results[entity_type].append(entity_text)

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