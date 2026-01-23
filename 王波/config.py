# -*- coding: utf-8 -*-

"""
配置参数信息
"""

"""
BERT模型:
BERT-base-chinese：通常有 12层 Transformer编码器
每层包含：
    多头自注意力机制（Multi-Head Self-Attention）
    前馈神经网络（Feed-Forward Network）
    层归一化（LayerNorm）
    残差连接（Residual Connections）
    
LoRA适配器层:
每个Transformer层中的：
    Query投影（1个LoRA适配器）
    Key投影（1个LoRA适配器）
    Value投影（1个LoRA适配器）
    输出投影（1个LoRA适配器）
总计：12层 × 4个适配器 = 48个LoRA适配器


顶部分类网络：
self.fc1 = nn.Linear(bert_hidden_size, hidden_size)        # 第1层
self.fc2 = nn.Linear(hidden_size, hidden_size // 2)       # 第2层
self.classify = nn.Linear(hidden_size // 2, class_num)    # 第3层
 3个全连接层
 2个LayerNorm层
 3个Dropout层
 CRF层（如果启用）
 
 
===总层数统计===
组件                  层数                 说明
BERT Transformer层    12	              BERT-base的标准配置
LoRA适配器	          48	              每层的Q、K、V、输出投影
全连接层	              3	                  fc1, fc2, classify
LayerNorm	          2	                  层归一化
Dropout	              3	                  防止过拟合
总计（近似）	         68层	              包括所有可训练模块
"""

Config = {
    "model_path": "model_output",  # 模型保存的目录路径
    "ori_model_path":r"E:\AI_Pycharm\_code\pythonProject\MyCode_9\ner_with_Bert\model_output\epoch_20.pth",
    "schema_path": "ner_data/schema.json",  # 标签映射文件路径
    "train_data_path": "ner_data/train",  # 训练数据文件路径
    "valid_data_path": "ner_data/test",  # 验证/测试数据文件路径
    "vocab_path": "chars.txt",  # 字符词汇表文件路径
    "max_length": 128,  # 输入序列的最大长度
    "hidden_size": 384,  # BERT输出后的全连接层隐藏层大小
    "num_layers": 3,  # 在BERT之上添加的全连接层层数
    "epoch": 80,  # 训练轮数
    "batch_size": 8,  # 批量大小，一次前向/反向传播处理的样本数量
    "optimizer": "AdamW",  # 优化器算法选择
    "learning_rate": 1e-5,  # 学习率，LoRA通常需要更小的学习率
    "use_crf": True,  # 是否使用条件随机场（CRF）层
    "class_num": 9,  # 分类类别数量
    "bert_path": r"E:\pretrain_models\bert-base-chinese",  # BERT路径

    # LoRA相关配置
    "use_lora": True,  # 是否使用LoRA
    "lora_r": 32,  # LoRA秩
    "lora_alpha": 64,  # LoRA的缩放系数，控制LoRA更新对原始参数的相对重要性，最终权重更新 = 原始权重 + (alpha/r) * LoRA更新
    "lora_dropout": 0.1,  # LoRA层的Dropout率，防止过拟合
    "lora_target_modules": ["query", "key", "value"],  # 对哪些模块应用LoRA
}