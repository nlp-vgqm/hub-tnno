# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import torch

from config import Config
from model import TorchModel, build_optimizer
from loader import load_data
from evaluate import Evaluator

import multiprocessing as mp
mp.set_start_method("spawn", force=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
LoRA + BERT NER 训练主程序
"""


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, tokenizer, config, label2id, id2label, epoch: int):
    save_dir = os.path.join(config["model_path"], f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    # peft 模型：保存 adapter（更轻）
    if hasattr(model, "backbone") and hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(save_dir)
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    tokenizer.save_pretrained(save_dir)

    # 保存 emission/CRF 头部参数
    if hasattr(model, "save_head"):
        model.save_head(save_dir)

    import json
    with open(os.path.join(save_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

    logger.info(f"模型已保存到: {save_dir}")


def main(config):
    set_seed(config.get("seed", 42))

    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据 & tokenizer & schema
    train_loader, train_dataset, tokenizer, label2id, id2label = load_data(config["train_data_path"], config, shuffle=True)

    # 有些配置从 schema 自动推断
    config["class_num"] = len(label2id)

    # 加载模型（BERT + LoRA）
    model = TorchModel(config, id2label=id2label, label2id=label2id)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("gpu可以使用，迁移模型至gpu")
    model = model.to(device)

    optimizer = build_optimizer(config, model)

    # 评估器（dev）
    evaluator = Evaluator(config, model, logger, tokenizer, label2id, id2label)

    use_amp = bool(config.get("fp16", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    global_step = 0

    for epoch in range(1, config["epoch"] + 1):
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs["loss"]

            scaler.scale(loss).backward()

            # 梯度裁剪
            max_norm = float(config.get("max_grad_norm", 1.0))
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())
            global_step += 1

            if step % max(1, int(len(train_loader) / 2)) == 0:
                logger.info("step %d/%d loss %f" % (step, len(train_loader), float(loss.item())))

        logger.info("epoch average loss: %f" % float(np.mean(train_loss)))

        evaluator.eval(epoch)
        save_model(model, tokenizer, config, label2id, id2label, epoch)

    return model


if __name__ == "__main__":
    main(Config)
