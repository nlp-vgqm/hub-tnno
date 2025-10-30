# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的RNN网络编写
实现一个NLP任务：判断特定字符在文本中的位置
特定字符在文本的第几位就属于哪一类（例如：qwerty文本，若特定字符指定为：e，那e在第3位就属于第3类）
使用交叉熵损失函数实现
"""


class TorchRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 嵌入层
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)  # RNN层
        self.classifier = nn.Linear(hidden_size, num_classes)  # 分类层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, target_char_idx, y=None):
        # 将目标字符的索引转换为与输入序列相同的形状
        target_char_embed = self.embedding(target_char_idx).unsqueeze(1)  # (batch_size, 1, embedding_dim)

        # 将目标字符嵌入与输入序列拼接
        x_embed = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # 将目标字符信息融入输入序列
        # 方法：将目标字符嵌入重复到与序列长度相同，然后与输入序列相加
        batch_size, seq_len, embedding_dim = x_embed.shape
        target_char_repeated = target_char_embed.repeat(1, seq_len, 1)  # (batch_size, seq_len, embedding_dim)
        combined_input = x_embed + target_char_repeated  # 将目标字符信息融入

        # RNN处理
        rnn_out, _ = self.rnn(combined_input)  # (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出作为整个序列的表示
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        # 分类
        y_pred = self.classifier(last_hidden)  # (batch_size, num_classes)

        if y is not None:
            return self.loss(y_pred, y)  # 计算损失
        else:
            return y_pred  # 返回预测结果


# 构建字符表
def build_vocab():
    # 包含所有可能的小写字母和特殊字符
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for index, char in enumerate(chars):
        vocab[char] = index + 2  # 从2开始编号，0和1留给特殊字符
    return vocab


# 生成样本
def build_sample(vocab, seq_length, target_char):
    # 随机生成一个文本序列
    chars = list(vocab.keys())[2:]  # 排除特殊字符
    seq = [random.choice(chars) for _ in range(seq_length)]

    # 查找目标字符在序列中的位置
    if target_char in seq:
        # 找到目标字符第一次出现的位置（从1开始计数）
        position = seq.index(target_char) + 1  # 位置从1开始
    else:
        # 如果目标字符不在序列中，标记为0类
        position = 0

    # 将字符序列转换为索引
    x = [vocab.get(char, vocab["[UNK]"]) for char in seq]
    target_char_idx = vocab.get(target_char, vocab["[UNK]"])

    return x, target_char_idx, position


# 建立数据集
def build_dataset(sample_num, vocab, seq_length, target_char):
    dataset_x = []
    dataset_target_char = []
    dataset_y = []

    for _ in range(sample_num):
        x, target_char_idx, y = build_sample(vocab, seq_length, target_char)
        dataset_x.append(x)
        dataset_target_char.append(target_char_idx)
        dataset_y.append(y)

    return (
        torch.LongTensor(dataset_x),
        torch.LongTensor(dataset_target_char),
        torch.LongTensor(dataset_y)
    )


# 评估模型
def evaluate(model, vocab, seq_length, target_char):
    model.eval()
    x, target_char_idx, y = build_dataset(200, vocab, seq_length, target_char)

    # 统计各类别的样本数量
    class_counts = [0] * (seq_length + 1)  # 0类 + 1到seq_length类
    for label in y:
        class_counts[label] += 1

    print("各类别样本数量：", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x, target_char_idx)
        _, predicted = torch.max(y_pred, 1)

        for y_p, y_t in zip(predicted, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 批次大小
    train_sample_num = 1000  # 训练样本数量
    test_sample_num = 200  # 测试样本数量
    seq_length = 6  # 序列长度
    embedding_dim = 16  # 嵌入维度
    hidden_size = 32  # 隐藏层大小
    num_classes = seq_length + 1  # 类别数：0类(不存在) + 1到seq_length类
    learning_rate = 0.001  # 学习率
    target_char = 'e'  # 目标字符

    # 建立字符表
    vocab = build_vocab()
    vocab_size = len(vocab)

    print("字符表：", vocab)
    print("目标字符：", target_char)
    print("类别数：", num_classes)

    # 建立模型
    model = TorchRNNModel(vocab_size, embedding_dim, hidden_size, num_classes)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 生成训练数据
        x, target_char_idx, y = build_dataset(train_sample_num, vocab, seq_length, target_char)

        # 分批训练
        for i in range(0, train_sample_num, batch_size):
            batch_x = x[i:i + batch_size]
            batch_target_char = target_char_idx[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            loss = model(batch_x, batch_target_char, batch_y)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            watch_loss.append(loss.item())

        # 打印训练信息
        avg_loss = np.mean(watch_loss)
        print(f"第{epoch + 1}轮平均loss: {avg_loss:.6f}")

        # 评估模型
        accuracy = evaluate(model, vocab, seq_length, target_char)
        log.append([accuracy, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "rnn_position_model.pth")

    # 保存字符表
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_num + 1), [l[0] for l in log], 'b-', label="准确率")
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.title("训练准确率变化")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch_num + 1), [l[1] for l in log], 'r-', label="损失值")
    plt.xlabel("训练轮次")
    plt.ylabel("损失值")
    plt.title("训练损失变化")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model, vocab


# 使用训练好的模型进行预测
def predict(model, vocab, input_strings, target_char):
    model.eval()

    # 将输入字符串转换为索引序列
    x_list = []
    target_char_idx_list = []

    for s in input_strings:
        # 将字符串转换为索引序列
        seq_indices = [vocab.get(c, vocab["[UNK]"]) for c in s]
        # 如果序列长度不足，用0填充
        if len(seq_indices) < 6:
            seq_indices += [vocab["[PAD]"]] * (6 - len(seq_indices))
        # 如果序列长度超过，截断
        elif len(seq_indices) > 6:
            seq_indices = seq_indices[:6]

        x_list.append(seq_indices)
        target_char_idx_list.append(vocab.get(target_char, vocab["[UNK]"]))

    x = torch.LongTensor(x_list)
    target_char_idx = torch.LongTensor(target_char_idx_list)

    with torch.no_grad():
        y_pred = model(x, target_char_idx)
        _, predicted = torch.max(y_pred, 1)

    for i, s in enumerate(input_strings):
        position = predicted[i].item()
        if position == 0:
            print(f"文本: '{s}', 目标字符 '{target_char}' 未出现")
        else:
            print(f"文本: '{s}', 目标字符 '{target_char}' 出现在第 {position} 位")


if __name__ == "__main__":
    # 训练模型
    trained_model, vocab_dict = main()

    # 测试预测
    test_strings = ["hello", "world", "example", "test", "python", "abcdef"]
    predict(trained_model, vocab_dict, test_strings, 'e')
