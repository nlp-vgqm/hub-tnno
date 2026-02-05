# -*- coding: utf-8 -*-
"""
模型效果测试（LoRA + BERT NER 版本）

- 从 labels / pred_labels（token-level）恢复到 char-level（word-level）实体
- 计算 macro/micro F1（按实体匹配）
"""

import torch
import numpy as np
from collections import defaultdict
from loader import load_data


def extract_entities(chars, label_ids, id2label):
    """从 BIO 标签抽取实体：返回 dict[type] -> list[str]"""
    results = defaultdict(list)
    cur = []
    cur_type = None

    for ch, lid in zip(chars, label_ids):
        lab = id2label.get(int(lid), "O")
        if lab.startswith("B-"):
            if cur:
                results[cur_type].append("".join(cur))
            cur = [ch]
            cur_type = lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur.append(ch)
        else:
            if cur:
                results[cur_type].append("".join(cur))
            cur, cur_type = [], None

    if cur:
        results[cur_type].append("".join(cur))
    return results


class Evaluator:
    def __init__(self, config, model, logger, tokenizer, label2id, id2label):
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = id2label

        self.valid_loader, self.valid_dataset, *_ = load_data(config["valid_data_path"], config, shuffle=False)

    @torch.no_grad()
    def eval(self, epoch: int):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        device = next(self.model.parameters()).device
        self.model.eval()

        # 统计：按实体类型
        stats = defaultdict(lambda: defaultdict(int))

        for batch_idx, batch in enumerate(self.valid_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            preds = outputs["predictions"].detach().cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            input_ids = batch["input_ids"].cpu().numpy()

            # 逐样本处理：用 tokenizer 的 word_ids 对齐回 char-level（只取每个字的第一个 subword）
            for i in range(input_ids.shape[0]):
                enc = self.tokenizer(
                    self.valid_dataset.sentences[batch_idx * self.config["batch_size"] + i],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=self.config["max_length"],
                )
                word_ids = enc.word_ids()

                char_pred = []
                char_true = []
                seen_word = set()
                for pos, wid in enumerate(word_ids):
                    if wid is None or wid in seen_word:
                        continue
                    seen_word.add(wid)
                    # 找到该 wid 对应的第一个 token 位置 pos
                    # labels/preds 的 pos 可能越界：因为我们上面 enc 是重新 tokenize 的，长度应一致（同 config）
                    if pos >= preds.shape[1]:
                        break
                    true_id = int(labels[i, pos])
                    pred_id = int(preds[i, pos])
                    # true_id 可能是 -100（special token），跳过
                    if true_id == -100:
                        continue
                    char_true.append(true_id)
                    char_pred.append(pred_id)

                chars = self.valid_dataset.sentences[batch_idx * self.config["batch_size"] + i]
                # 截断到一致长度
                n = min(len(chars), len(char_true), len(char_pred))
                chars = chars[:n]
                char_true = char_true[:n]
                char_pred = char_pred[:n]

                true_entities = extract_entities(chars, char_true, self.id2label)
                pred_entities = extract_entities(chars, char_pred, self.id2label)

                # 关注四类（与原作业一致）
                for ent_type in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                    stats[ent_type]["正确识别"] += len([e for e in pred_entities.get(ent_type, []) if e in true_entities.get(ent_type, [])])
                    stats[ent_type]["样本实体数"] += len(true_entities.get(ent_type, []))
                    stats[ent_type]["识别出实体数"] += len(pred_entities.get(ent_type, []))

        self._show(stats)

    def _show(self, stats):
        F1_scores = []
        for ent_type in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            correct = stats[ent_type]["正确识别"]
            pred_n = stats[ent_type]["识别出实体数"]
            true_n = stats[ent_type]["样本实体数"]

            precision = correct / (pred_n + 1e-5)
            recall = correct / (true_n + 1e-5)
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(f1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (ent_type, precision, recall, f1))

        self.logger.info("Macro-F1: %f" % float(np.mean(F1_scores)))

        correct_all = sum(stats[t]["正确识别"] for t in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])
        pred_all = sum(stats[t]["识别出实体数"] for t in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])
        true_all = sum(stats[t]["样本实体数"] for t in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])

        micro_p = correct_all / (pred_all + 1e-5)
        micro_r = correct_all / (true_all + 1e-5)
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
