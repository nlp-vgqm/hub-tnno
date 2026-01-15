# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertConfig, BertModel, BertForMaskedLM

"""
基于BERT的自回归语言模型
"""


class BERTLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len):
        super(BERTLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 创建BERT配置（使用解码器风格，有因果注意力）
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            max_position_embeddings=max_seq_len,
            is_decoder=True,  # 设置为解码器，实现自回归
            is_encoder_decoder=False,
        )

        # 使用BERT模型
        self.bert = BertModel(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 生成因果注意力掩码（上三角矩阵）
    def generate_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, attention_mask=None):
        batch_size, seq_len = x.shape

        # 生成因果注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len)

        # 生成因果掩码（确保看不到未来的token）
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)  # [batch, 1, seq_len, seq_len]

        # 组合注意力掩码
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 添加因果掩码
        extended_attention_mask = extended_attention_mask + causal_mask * -10000.0

        # BERT前向传播
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        sequence_output = self.dropout(sequence_output)
        y_pred = self.lm_head(sequence_output)  # [batch, seq_len, vocab_size]

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    vocab["<UNK>"] = len(vocab)  # 确保有UNK token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
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

    # 创建注意力掩码（非padding位置为1）
    attention_masks = []
    for x in dataset_x:
        mask = [1] * len(x)
        attention_masks.append(mask)

    return (
        torch.LongTensor(dataset_x),
        torch.LongTensor(dataset_y),
        torch.FloatTensor(attention_masks)
    )


# 建立模型
def build_model(vocab, char_dim, max_seq_len):
    vocab_size = len(vocab)
    model = BERTLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=char_dim,
        num_layers=6,  # BERT-base通常用12层，这里用6层减少计算量
        num_heads=8,  # 注意力头数
        max_seq_len=max_seq_len
    )
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        generated = openings

        # 生成了换行符，或生成文本超过50字则终止迭代
        while pred_char != "\n" and len(generated) <= 50:
            current_text = generated[-window_size:] if len(generated) > window_size else generated
            x = [vocab.get(char, vocab["<UNK>"]) for char in current_text]

            # 填充到窗口大小
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x

            x = torch.LongTensor([x])
            attention_mask = torch.FloatTensor([[1 if token != vocab["<pad>"] else 0 for token in x[0]]])

            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()

            y = model(x, attention_mask=attention_mask)[0][-1]  # 取最后一个位置的预测
            index = sampling_strategy(y)
            pred_char = reverse_vocab.get(index, "<UNK>")
            generated += pred_char

    return generated


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        # 确保概率分布有效
        prob_distribution = np.maximum(prob_distribution, 1e-10)
        prob_distribution = prob_distribution / prob_distribution.sum()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


# 计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]

            # 填充到窗口大小
            if len(x) < window_size:
                x = [vocab["<pad>"]] * (window_size - len(x)) + x

            x = torch.LongTensor([x])
            attention_mask = torch.FloatTensor([[1 if token != vocab["<pad>"] else 0 for token in x[0]]])

            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])

            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()

            pred_prob_distribute = model(x, attention_mask=attention_mask)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob + 1e-10, 10)  # 避免log(0)

    if prob == 0:
        return float('inf')

    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 减少batch_size，因为BERT需要更多显存
    train_sample = 50000  # 每轮训练总共训练的样本总数
    char_dim = 256  # 每个字的维度
    window_size = 20  # 增加窗口大小，BERT能处理更长序列
    max_seq_len = 64  # 最大序列长度

    vocab = build_vocab("vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, char_dim, max_seq_len)  # 建立模型

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 使用AdamW优化器，更适合BERT
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch_num)  # 学习率调度

    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            x, y, attention_masks = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
                attention_masks = attention_masks.cuda()

            optim.zero_grad()  # 梯度归零
            loss = model(x, y, attention_mask=attention_masks)  # 计算loss
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        scheduler.step()  # 更新学习率

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print("学习率:", optim.param_groups[0]['lr'])

        # 生成测试
        print("生成示例1:", generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print("生成示例2:", generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

        # 计算困惑度示例
        test_sentence = "今天天气很好，我们出去散步吧。"
        ppl = calc_perplexity(test_sentence, model, vocab, window_size)
        print(f"测试句子的困惑度: {ppl:.2f}")

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'config': {
                'char_dim': char_dim,
                'window_size': window_size,
                'max_seq_len': max_seq_len
            }
        }, model_path)
        print(f"模型已保存到: {model_path}")
        return


if __name__ == "__main__":
    train("corpus.txt", False)
