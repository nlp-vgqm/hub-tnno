# -*- coding: utf-8 -*-
"""\
预测脚本（LoRA + BERT + CRF NER）

用法：
1) 训练后会在 model_output/epoch_X 下保存：
   - LoRA adapter（PEFT）
   - tokenizer
   - labels.json
   - head.pt（emission + CRF 参数）
2) 修改下面 model_dir 指向某个 epoch 目录，然后运行：
   python predict.py
"""

import os
import json
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftConfig, PeftModel

from model import TorchModel


def load_labels(model_dir: str):
    with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)

    # 兼容 json key 为字符串的情况
    if isinstance(next(iter(labels["id2label"].keys())), str):
        id2label = {int(k): v for k, v in labels["id2label"].items()}
    else:
        id2label = labels["id2label"]
    label2id = labels["label2id"]
    return label2id, id2label


def load_model(model_dir: str):
    label2id, id2label = load_labels(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # 尝试读取 peft adapter config（获取 base model 名称）
    try:
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base_name = peft_cfg.base_model_name_or_path
        base = AutoModel.from_pretrained(base_name)
        backbone = PeftModel.from_pretrained(base, model_dir)
    except Exception:
        # 如果不是 adapter（不推荐），退化为直接从目录加载 encoder
        base = AutoModel.from_pretrained(model_dir)
        backbone = base

    cfg = {
        "pretrained_model": "__dummy__",  # 不会被真正使用（我们会覆盖 backbone）
        "use_lora": False,
        "use_crf": True,
    }
    model = TorchModel(cfg, id2label=id2label, label2id=label2id)
    model.backbone = backbone
    model.load_head(model_dir)

    return model, tokenizer, id2label


def extract_entities(chars: List[str], labels: List[str]) -> List[Tuple[str, str]]:
    ents = []
    cur = []
    cur_type = None
    for ch, lab in zip(chars, labels):
        if lab.startswith("B-"):
            if cur:
                ents.append((cur_type, "".join(cur)))
            cur = [ch]
            cur_type = lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur.append(ch)
        else:
            if cur:
                ents.append((cur_type, "".join(cur)))
            cur, cur_type = [], None
    if cur:
        ents.append((cur_type, "".join(cur)))
    return ents


@torch.no_grad()
def predict(model, tokenizer, id2label, text: str, max_length: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chars = list(text.strip())
    enc = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = tokenizer(chars, is_split_into_words=True, truncation=True, max_length=max_length).word_ids()

    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    pred_ids = outputs["predictions"].squeeze(0).detach().cpu().tolist()

    # 取每个字的第一个 subword 的预测
    char_pred = []
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        if pos >= len(pred_ids):
            break
        char_pred.append(id2label[int(pred_ids[pos])])

    char_pred = char_pred[:len(chars)]
    ents = extract_entities(chars, char_pred)
    return char_pred, ents


if __name__ == "__main__":
    model_dir = os.path.join("model_output", "epoch_1")  # 改成你要用的 epoch
    model, tokenizer, id2label = load_model(model_dir)

    text = "中国政府对目前南亚出现的核军备竞赛的局势深感忧虑和不安"
    labels, ents = predict(model, tokenizer, id2label, text)
    print(text)
    print("".join([f"{c}/{l} " for c, l in zip(list(text), labels)]))
    print("Entities:", ents)
