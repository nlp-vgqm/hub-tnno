#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertConfig
from typing import Optional

"""
基于pytorch + BERT的自回归语言模型
核心改动：用BERT替换LSTM，添加下三角掩码实现自回归
"""

class BertLanguageModel(nn.Module):
    def __init__(self, vocab_size, char_dim, dropout_rate=0.1):
        super(BertLanguageModel, self).__init__()
        # 初始化BERT配置（轻量级配置适配字符级任务）
        self.bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=char_dim,
            num_hidden_layers=2,  # 减小层数降低计算量
            num_attention_heads=4,
            intermediate_size=char_dim*4,
            max_position_embeddings=128,  # 适配最大序列长度
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            pad_token_id=0
        )
        
        # 加载BERT模型（随机初始化，适配自定义字表）
        self.bert = BertModel(self.bert_config)
        # 分类层：将BERT输出映射到词汇表大小
        self.classify = nn.Linear(char_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = nn.functional.cross_entropy

    # 生成自回归掩码（下三角矩阵，保证每个token只能看到前面的token）
    def _generate_autoregressive_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 生成下三角掩码：1表示可见，0表示不可见
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        # 将0替换为-10000（BERT的attention mask要求：-inf表示mask）
        mask = mask.masked_fill(mask == 0, -10000.0)
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask

    # x: [batch_size, seq_len]
    # y: [batch_size, seq_len] (可选，训练时传入)
    def forward(self, x, y=None):
        batch_size, seq_len = x.shape
        device = x.device
        
        # 生成padding mask（标记pad token的位置）
        padding_mask = (x != 0).long()
        
        # 生成自回归注意力掩码
        autoregressive_mask = self._generate_autoregressive_mask(seq_len, device)
        
        # BERT前向传播
        bert_output = self.bert(
            input_ids=x,
            attention_mask=padding_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            # 关键：传入自回归掩码
            attention_mask=padding_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            cross_attn_head_mask=None,
            inputs_embeds=None,
            # BERT的自注意力掩码（下三角）
            encoder_attn_mask=autoregressive_mask
        )
        
        # 获取BERT最后一层输出 [batch_size, seq_len, char_dim]
        hidden_states = bert_output.last_hidden_state
        hidden_states = self.dropout(hidden_states)  # 使用Dropout正则化
        y_pred = self.classify(hidden_states)        # [batch_size, seq_len, vocab_size]
        
        if y is not None:
            # 训练时计算损失：展平序列维度
            return self.loss(y_pred.reshape(-1, y_pred.shape[-1]), y.reshape(-1))
        else:
            # 预测时返回概率分布
            return torch.softmax(y_pred, dim=-1)

# 加载字表（修复<UNK>缺失问题）
def build_vocab(vocab_path):
    vocab = {"<pad>":0, "<UNK>":1}  # 补充<UNK>字符，索引1
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()  # 更鲁棒的换行符处理
            if char:  # 跳过空行
                vocab[char] = index + 2  # 0:pad, 1:unk, 从2开始分配
    return vocab

# 加载语料（修复编码问题，兼容utf8/gbk）
def load_corpus(path):
    corpus = ""
    encodings = ["utf8", "gbk", "gb2312"]
    for encoding in encodings:
        try:
            with open(path, encoding=encoding) as f:
                for line in f:
                    corpus += line.strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法解码文件 {path}，请检查文件编码")
    return corpus

# 随机生成一个样本（逻辑与原代码一致）
def build_sample(vocab, window_size, corpus):
    # 保证样本不越界
    if len(corpus) <= window_size + 1:
        return build_sample(vocab, window_size, corpus)
    
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位（自回归核心）
    
    # 将字转换成序号（使用<UNK>处理未登录字）
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

# 建立数据集（逻辑与原代码一致）
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立BERT语言模型
def build_model(vocab, char_dim):
    model = BertLanguageModel(len(vocab), char_dim)
    return model

# 文本生成测试代码（适配BERT模型）
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成换行符/超过30字则终止
        while pred_char != "\n" and len(openings) <= 30:
            # 拼接上一轮预测的字符
            if pred_char:
                openings += pred_char
            
            # 截取最后window_size个字符作为输入
            input_chars = openings[-window_size:] if len(openings) >= window_size else openings
            # 补pad到window_size长度
            x = [vocab.get(char, vocab["<UNK>"]) for char in input_chars]
            x = [0]*(window_size - len(x)) + x  # 左侧补pad
            
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 预测下一个字符（取最后一个位置的输出）
            y_prob = model(x)[0][-1]
            index = sampling_strategy(y_prob)
            
            # 转换回字符
            pred_char = reverse_vocab.get(index, "<UNK>")
    return openings

# 采样策略（与原代码一致）
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        # 归一化概率（避免数值误差导致和不为1）
        prob_distribution = prob_distribution / np.sum(prob_distribution)
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

# 计算文本ppl（逻辑与原代码一致）
def calc_perplexity(sentence, model, vocab, window_size):
    if len(sentence) <= 1:
        return float("inf")
    
    prob = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            
            # 构建输入（补pad）
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = [0]*(window_size - len(x)) + x if len(x) < window_size else x[-window_size:]
            x = torch.LongTensor([x])
            
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            
            if torch.cuda.is_available():
                x = x.cuda()
            
            # 预测概率
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index].item()
            
            # 防止log(0)
            if target_prob <= 1e-10:
                target_prob = 1e-10
            prob += math.log(target_prob, 10)
    
    # 计算PPL
    ppl = 2 ** (prob * (-1 / len(sentence)))
    return ppl

# 训练函数（优化学习率等参数）
def train(corpus_path, vocab_path="vocab.txt", save_weight=True):
    # 训练参数（适配BERT调整）
    epoch_num = 10                # BERT收敛更快，减少训练轮数
    batch_size = 32               # BERT计算量大，减小批次大小
    train_sample = 20000          # 每轮训练样本数
    char_dim = 256                # BERT隐藏层维度
    window_size = 10              # 序列窗口大小
    lr = 1e-4                     # BERT建议使用更小的学习率
    
    # 加载字表和语料
    print("加载字表...")
    vocab = build_vocab(vocab_path)
    print(f"字表大小：{len(vocab)}")
    
    print("加载语料...")
    corpus = load_corpus(corpus_path)
    print(f"语料长度：{len(corpus)} 字符")
    
    # 建立模型
    print("初始化BERT语言模型...")
    model = build_model(vocab, char_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备：{device}")
    
    # 优化器（添加权重衰减）
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.8)
    
    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch in range(int(train_sample / batch_size)):
            # 构建批次数据
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            x, y = x.to(device), y.to(device)
            
            # 梯度归零
            optim.zero_grad()
            # 计算损失
            loss = model(x, y)
            # 反向传播
            loss.backward()
            # 梯度裁剪（防止BERT训练梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新参数
            optim.step()
            
            watch_loss.append(loss.item())
            
            # 打印批次损失（每100批次）
            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epoch_num}, Batch {batch+1}, Loss: {np.mean(watch_loss[-100:]):.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 打印本轮平均损失
        avg_loss = np.mean(watch_loss)
        print("="*50)
        print(f"第{epoch+1}轮训练完成，平均Loss: {avg_loss:.4f}")
        
        # 生成测试文本
        print("测试文本生成：")
        test_openings = ["让他在半年之前，就不能做出", "李慕站在山路上，深深的呼吸"]
        for opening in test_openings:
            gen_sent = generate_sentence(opening, model, vocab, window_size)
            print(f"开头：{opening} → 生成：{gen_sent}")
        
        # 计算测试文本PPL
        test_sent = "李慕站在山路上，深深的呼吸着新鲜空气"
        ppl = calc_perplexity(test_sent, model, vocab, window_size)
        print(f"测试文本PPL：{ppl:.2f}")
        print("="*50)
    
    # 保存模型
    if save_weight:
        os.makedirs("model", exist_ok=True)
        base_name = os.path.basename(corpus_path).replace(".txt", "_bert.pth")
        model_path = os.path.join("model", base_name)
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "char_dim": char_dim,
            "window_size": window_size
        }, model_path)
        print(f"模型已保存至：{model_path}")

if __name__ == "__main__":
    # 注意：需要提前准备vocab.txt（每行一个字符）和corpus.txt（训练语料）
    train("corpus.txt", vocab_path="vocab.txt", save_weight=True)
