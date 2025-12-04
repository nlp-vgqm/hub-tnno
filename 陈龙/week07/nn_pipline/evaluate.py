# -*- coding: utf-8 -*-
import torch
from loader import load_data
import time

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.total_samples = len(self.valid_data)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果

        self.total_time = 0

        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data

            # 开始计时
            start_time = time.time()
            with torch.no_grad():
                pred_results = self.model(input_ids)
            batch_time = time.time() - start_time

            self.total_time += batch_time

            self.write_stats(labels, pred_results)
        acc,avg_time_per_100 = self.show_stats()
        return acc, avg_time_per_100

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("平均每条预测耗时：%f ms" % (self.total_time / self.total_samples * 1000))
        self.logger.info("--------------------")
        return correct / (correct + wrong), self.total_time / self.total_samples * 100 * 1000
