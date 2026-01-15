# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertConfig, BertModel, BertTokenizer, BertForMaskedLM

"""
基于Bert的自回归语言模型，使用mask机制实现单向生成
"""


class BertARModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4):
        super(BertARModel, self).__init__()

        # 创建Bert配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=512,
            type_vocab_size=1,
            is_decoder=True,  # 关键设置：作为解码器
        )

        # 使用BertModel
        self.bert = BertModel(config)

        # 添加语言模型头
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 初始化语言模型头
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # 初始化Bert的权重
        for module in self.bert.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        """
        前向传播
        Args:
            x: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码
        """
        batch_size, seq_len = x.shape

        # 创建padding mask（全1）
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=x.device)

        # 调用Bert模型
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return logits

    def compute_loss(self, x, y):
        """计算损失"""
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y.view(-1),
            ignore_index=0  # 忽略padding token
        )
        return loss

    def predict_next(self, x):
        """预测下一个token的概率分布"""
        logits = self.forward(x)
        # 取最后一个位置的预测
        next_token_logits = logits[:, -1, :]
        return torch.softmax(next_token_logits, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    # 添加特殊token
    vocab["<UNK>"] = len(vocab)
    vocab["<EOS>"] = len(vocab)  # 结束符
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim=256, num_layers=4, num_heads=4):
    model = BertARModel(len(vocab), char_dim, num_layers, num_heads)
    return model


# 文本生成测试代码（自回归生成）
def generate_sentence(openings, model, vocab, window_size, max_len=50):
    """生成文本"""
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()

    with torch.no_grad():
        current_text = openings

        # 将开头文本编码为token ids
        input_chars = list(openings)
        if len(input_chars) > window_size:
            input_chars = input_chars[-window_size:]

        input_ids = [vocab.get(char, vocab["<UNK>"]) for char in input_chars]

        # 生成文本
        for i in range(max_len):
            # 准备输入
            x = torch.LongTensor([input_ids])

            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()

            # 预测下一个token
            y_pred = model.predict_next(x)

            # 采样策略
            probs = y_pred[0].cpu().numpy()

            # 应用温度参数
            temperature = 0.8
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / probs.sum()

            # 只从概率最高的前10个token中采样
            top_k = 10
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / top_probs.sum()

            # 采样
            index = np.random.choice(top_indices, p=top_probs)

            # 获取对应的字符
            next_char = reverse_vocab.get(index, "")

            # 如果生成了结束符或特殊token，停止生成
            if next_char in ["", "<UNK>", "<pad>", "<EOS>"]:
                break

            # 添加到文本中
            current_text += next_char

            # 更新输入
            input_ids.append(index)
            if len(input_ids) > window_size:
                input_ids = input_ids[-window_size:]

        return current_text


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 10  # 样本文本长度
    num_layers = 4  # Bert层数
    num_heads = 4  # 注意力头数

    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, char_dim, num_layers, num_heads)  # 建立模型

    print(f"词表大小: {len(vocab)}")

    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU训练")
    else:
        print("使用CPU训练")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    print("开始训练...")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model.compute_loss(x, y)  # 计算loss
            loss.backward()  # 计算梯度

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()  # 更新权重
            watch_loss.append(loss.item())

            if batch % 200 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))

        # 生成测试
        model.eval()
        print("生成示例1:")
        try:
            result1 = generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size)
            print(result1)
        except Exception as e:
            print(f"生成失败: {e}")

        print("生成示例2:")
        try:
            result2 = generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size)
            print(result2)
        except Exception as e:
            print(f"生成失败: {e}")

        # 打印词表大小和模型信息
        print(f"词表大小: {len(vocab)}")
        print(f"输入示例: {list('让他在半年之前，就不能做出')}")
        print(f"输入长度: {window_size}")

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)

        # 保存模型
        save_dict = {
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'config': {
                'hidden_size': char_dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'window_size': window_size
            }
        }

        torch.save(save_dict, model_path)
        print(f"模型已保存到: {model_path}")
        return


if __name__ == "__main__":
    # 训练模型
    train("corpus.txt", False)
