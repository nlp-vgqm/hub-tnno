"""
基于seq2seq架构的SFT微调示例（使用T5模型）
"""

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import numpy as np
import evaluate

# 配置
MODEL_NAME = "google-t5/t5-small"  # 轻量版T5模型，适合演示
OUTPUT_DIR = "./output_seq2seq"
MAX_SOURCE_LENGTH = 128  # 输入文本最大长度
MAX_TARGET_LENGTH = 128  # 输出文本最大长度

# 加载评估指标（BLEU分数，适用于文本生成）
metric = evaluate.load("bleu")

def create_chinese_seq2seq_dataset():
    """构造中文seq2seq数据集（输入-输出格式）"""
    # 数据格式：input为问题，output为答案
    data = [
        {
            "input": "介绍一下人工智能",
            "output": "人工智能是计算机科学的分支，致力于创建能模拟人类智能的系统，包括学习、推理等能力。"
        },
        {
            "input": "什么是深度学习",
            "output": "深度学习是机器学习的子领域，使用多层神经网络学习数据的复杂模式，模仿人脑结构。"
        },
        {
            "input": "Python列表和元组的区别",
            "output": "列表是可变的，用[]表示；元组不可变，用()表示。"
        },
        {
            "input": "解释监督学习",
            "output": "监督学习是用标记数据训练模型，学习输入到输出的映射，用于预测新数据。"
        },
        {
            "input": "如何提高模型泛化能力",
            "output": "增加数据多样性、使用正则化、数据增强、交叉验证可提高模型泛化能力。"
        }
    ]
    return data

def preprocess_function(examples, tokenizer):
    """预处理数据：将输入和输出转换为模型可接受的格式"""
    inputs = [f"question: {item['input']}" for item in examples]  # 给输入加前缀，提升模型理解
    targets = [item['output'] for item in examples]

    # 编码输入
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 编码输出（用于计算损失）
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 将padding位置的label设为-100（忽略计算损失）
    labels = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
    model_inputs["labels"] = labels
    return model_inputs

def compute_metrics(eval_pred, tokenizer):
    """计算评估指标（BLEU分数）"""
    predictions, labels = eval_pred
    # 将预测结果解码为文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将标签中-100替换为pad_token，再解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算BLEU分数（适用于短文本）
    result = metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]  # BLEU要求参考文本是列表的列表
    )
    return {"bleu": result["bleu"]}

def load_model_and_tokenizer():
    """加载T5模型和分词器"""
    print(f"正在加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return model, tokenizer

def main():
    print("=" * 50)
    print("开始Seq2Seq SFT微调训练")
    print("=" * 50)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()

    # 创建并预处理数据集
    print("\n正在构造训练数据...")
    raw_data = create_chinese_seq2seq_dataset()
    dataset = Dataset.from_list(raw_data)
    # 应用预处理函数
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True  # 批量处理
    )
    print(f"训练样本数量: {len(tokenized_dataset)}")

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=5,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        evaluation_strategy="no",  # 示例数据少，不做验证
        predict_with_generate=True,  # 生成式评估
        fp16=False,
        remove_unused_columns=False
    )

    # 创建Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)  # 绑定评估函数
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    print(f"\n训练完成，保存模型到 {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 50)
    print("Seq2Seq SFT训练完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()