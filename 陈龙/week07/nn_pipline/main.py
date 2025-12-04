# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc, avg_time_per_100 = evaluator.eval(epoch)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, avg_time_per_100


def write_result(current_acc, avg_time_per_100, Config):
    # 记录结果
    result_file = "experiment_results.csv"

    # 准备要记录的数据
    result_data = {
        "model_type": Config["model_type"],
        "learning_rate": Config["learning_rate"],
        "hidden_size": Config["hidden_size"],
        "batch_size": Config["batch_size"],
        "pooling_style": Config["pooling_style"],
        "avg_time_per_100 (ms)": avg_time_per_100,
        "accuracy": current_acc,
    }

    # 检查文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(result_file):
        with open(result_file, "w", encoding="utf-8") as f:
            header = ",".join(result_data.keys()) + "\n"
            f.write(header)

    # 追加当前实验结果
    with open(result_file, "a", encoding="utf-8") as f:
        values = [str(value) for value in result_data.values()]
        line = ",".join(values) + "\n"
        f.write(line)

    logger.info(f"实验结果已记录到 {result_file}: 准确率={current_acc:.4f}")


if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    result_file = "experiment_results.csv"
    if os.path.exists(result_file):
        os.remove(result_file)
        logger.info("已删除旧结果文件")

    best_acc = 0
    for model in ['bert', "bert_lstm", "bert_cnn", "bert_mid_layer"]:
        # for model in ["fast_text", "gru", "cnn", "rnn", "gated_cnn", "stack_gated_cnn", "rcnn", 'bert', "bert_lstm",
        #           "bert_cnn", "bert_mid_layer"]:
        Config["model_type"] = model
        for lr in [1e-4, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        current_acc, avg_time_per_100 = main(Config)
                        print("本次配置准确率：", current_acc, "当前配置：", Config)

                        write_result(current_acc, avg_time_per_100, Config)

                        if current_acc > best_acc:
                            best_acc = current_acc
                            print("最优结果更新:", Config["model_type"], "准确率：", best_acc, "当前配置：", Config)
