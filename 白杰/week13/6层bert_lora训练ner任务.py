import os
import torch
import evaluate
import numpy as np
from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    BertConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import classification_report

# ===================== 1. 配置参数 =====================
# 基础配置
MODEL_NAME = "prajjwal1/bert-mini"  # 6层BERT，128维隐藏层
LABEL_ALL_TOKENS = True  # 是否对所有token标注（解决subword拆分问题）
MAX_LENGTH = 128  # 适配bert-mini的最大长度
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-4
LORA_R = 8  # LoRA秩
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 自定义NER标签（根据你的数据集修改！）
# 示例：BIO格式，需替换为你数据集的实际标签
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

# ===================== 2. 加载自定义NER数据集 =====================
def load_custom_ner_data(data_dir: str) -> DatasetDict:
    """
    加载自定义NER数据集（CONLL格式）
    数据集格式要求：
    - 每个文件中，每行是 "token label"（空格分隔）
    - 句子之间用空行分隔
    - 需包含train.txt、dev.txt、test.txt三个文件
    """
    def _parse_file(file_path: str) -> Dict[str, List[List[str]]]:
        tokens_list = []
        labels_list = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if current_tokens:
                        tokens_list.append(current_tokens)
                        labels_list.append(current_labels)
                        current_tokens = []
                        current_labels = []
                else:
                    # 拆分token和label（处理多个空格的情况）
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_tokens.append(token)
                        current_labels.append(label)
        
        # 处理最后一个句子
        if current_tokens:
            tokens_list.append(current_tokens)
            labels_list.append(current_labels)
        
        return {"tokens": tokens_list, "labels": labels_list}
    
    # 加载训练/验证/测试集
    train_data = _parse_file(os.path.join(data_dir, "train.txt"))
    dev_data = _parse_file(os.path.join(data_dir, "dev.txt"))
    test_data = _parse_file(os.path.join(data_dir, "test.txt"))
    
    # 转换为datasets格式
    dataset_dict = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(dev_data),
        "test": Dataset.from_dict(test_data)
    })
    return dataset_dict

# ===================== 3. 数据预处理 =====================
def preprocess_function(examples):
    """预处理函数：tokenize + 标签对齐（处理subword）"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # None表示特殊token（[CLS]/[SEP]/padding），标注为-100（训练时忽略）
            if word_idx is None:
                label_ids.append(-100)
            # 同一个word的subword，仅标注第一个token（或全部标注）
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[label[word_idx]])
            else:
                label_ids.append(LABEL2ID[label[word_idx]] if LABEL_ALL_TOKENS else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# ===================== 4. 评估指标 =====================
def compute_metrics(p):
    """计算NER任务评估指标（precision/recall/f1）"""
    metric = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 移除-100标签（特殊token）
    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# ===================== 5. 构建LoRA+BERT模型 =====================
def build_lora_bert_model():
    """构建结合LoRA的BERT模型"""
    # 加载基础模型
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True  # 适配标签数不一致的情况
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # 序列标注任务
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],  # BERT的attention层目标模块
        bias="none",
        inference_mode=False
    )
    
    # 融合LoRA到基础模型
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()  # 打印可训练参数（LoRA仅微调少量参数）
    return lora_model

# ===================== 6. 训练函数 =====================
def train_ner_model(data_dir: str, output_dir: str = "./lora_bert_ner"):
    """训练LoRA-BERT NER模型"""
    # 1. 加载数据
    dataset = load_custom_ner_data(data_dir)
    
    # 2. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    # 3. 数据预处理
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # 4. 数据collator（处理padding和标签）
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # 5. 构建模型
    model = build_lora_bert_model()
    model.to(DEVICE)
    
    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",  # 每个epoch评估一次
        save_strategy="epoch",        # 每个epoch保存一次
        load_best_model_at_end=True,  # 加载最优模型
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),  # 混合精度训练（GPU可用时）
        report_to="none"
    )
    
    # 7. 构建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 8. 开始训练
    trainer.train()
    
    # 9. 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 10. 测试集评估
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("测试集评估结果：", test_results)
    return model, tokenizer

# ===================== 7. 预测函数 =====================
def predict_ner(text: str, model, tokenizer):
    """
    NER预测函数
    :param text: 输入文本（字符串）
    :param model: 训练好的LoRA-BERT模型
    :param tokenizer: 对应的tokenizer
    :return: 标注结果（list of (token, label)）
    """
    # 预处理文本
    tokens = text.split()  # 按空格分词（可替换为你的分词逻辑）
    inputs = tokenizer(
        tokens,
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 解析预测结果
    word_ids = inputs.word_ids(batch_index=0)
    pred_labels = []
    previous_word_idx = None
    
    for word_idx, pred_idx in zip(word_ids, predictions[0].cpu().numpy()):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        pred_labels.append(ID2LABEL[pred_idx])
        previous_word_idx = word_idx
    
    # 对齐token和标签
    results = list(zip(tokens, pred_labels))
    return results

# ===================== 主函数（运行入口） =====================
if __name__ == "__main__":
    
    DATA_DIR = "./week9 序列标注问题/ner"
    
    # 1. 训练模型
    model, tokenizer = train_ner_model(DATA_DIR)
    
    # 2. 示例预测
    test_text = "张三在北京市百度科技有限公司工作"
    ner_results = predict_ner(test_text, model, tokenizer)
    print("NER预测结果：")
    for token, label in ner_results:
        print(f"{token}\t{label}")
    
    # 3. 加载保存的模型进行预测（训练后单独预测用）
    # base_model = AutoModelForTokenClassification.from_pretrained(
    #     MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    # )
    # lora_model = PeftModel.from_pretrained(base_model, "./lora_bert_ner")
    # predict_ner("测试文本", lora_model, tokenizer)
