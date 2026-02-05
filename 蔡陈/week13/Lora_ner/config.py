# -*- coding: utf-8 -*-
"""\
配置参数信息（LoRA + BERT + CRF NER 版本）

说明：
- Encoder: 预训练 Transformer（默认 bert-base-chinese）
- Emission: Linear(hidden -> num_labels)
- Decode: 线性链 CRF
- 可选开启 LoRA（PEFT），只训练少量 adapter 参数
"""

Config = {
    # 输出目录
    "model_path": "model_output",

    # 数据与标签
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/dev",
    "test_data_path": "ner_data/test",

    # 预训练模型（优先使用本地路径；也可写成 'bert-base-chinese' 等 HuggingFace 名称）
    # 例如："bert-base-chinese" 或者 "/path/to/bert-base-chinese"
    "pretrained_model": "bert-base-chinese",

    # 序列长度（注意：BERT 最大一般为 512）
    "max_length": 128,

    # 训练超参（BERT/CRF 常用学习率：1e-5~5e-5）
    "epoch": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,

    # LoRA 配置
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,

    # BERT 系列常用 target_modules（BertSelfAttention 的 query/key/value）
    # 不同 backbone 可能不同（如 LLaMA: q_proj/k_proj/v_proj）
    "lora_target_modules": ["query", "key", "value"],

    # CRF
    "use_crf": True,

    # 其他
    "seed": 42,
    "fp16": True,
    "num_workers": 0,

    # 类别数（从 schema.json 自动推断；这里保留是为了兼容旧代码）
    "class_num": 9,
}
