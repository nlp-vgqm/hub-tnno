#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json




def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}  # 补齐字符
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab["unk"] = len(vocab)  # 未知字符
    return vocab

def build_sample(vocab):
    # 生成随机文本
    x_chars = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 生成序列标签
    y = [1 if char in TARGET_CHARS else 0 for char in x_chars]
    # 字符转索引
    x = [vocab.get(char, vocab["unk"]) for char in x_chars]
    return x, y

def build_dataset(sample_num, vocab):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_num):
        x, y = build_sample(vocab)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


class RNNPositionModel(nn.Module):
    def __init__(self, vocab_size, vector_dim, hidden_size, num_layers):
        super(RNNPositionModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=vector_dim,
            padding_idx=0
        )
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入格式：[batch_size, seq_len, embed_dim]
        )
        self.classify = nn.Linear(hidden_size, 1)  # 每个位置输出1个预测值
        self.activation = torch.sigmoid
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x, y=None):
        embed_x = self.embedding(x)  # [batch_size, seq_len, vector_dim]
        out, _ = self.rnn(embed_x)   # [batch_size, seq_len, hidden_size]
        logits = self.classify(out)  # [batch_size, seq_len, 1]

        y_pred = self.activation(logits)
        if y is not None:
            y = y.unsqueeze(dim=2)  # 调整标签维度与预测一致
            return self.loss(logits, y)
        else:
            return y_pred

TARGET_CHARS = {"你", "我"}
vector_dim = 20
sentence_length = 6
hidden_size = 32
num_layers = 1
batch_size = 20
epoch_num = 15
train_sample = 800

learning_rate = 0.005

def evaluate(model, vocab):
    model.eval()
    test_x, test_y = build_dataset(sample_num=200, vocab=vocab)
    total_pos = test_x.shape[0] * test_x.shape[1]
    correct_pos = 0

    with torch.no_grad():
        y_pred = model(test_x)
        y_pred_np = y_pred.squeeze(dim=2).numpy()
        test_y_np = test_y.numpy()

        for i in range(test_x.shape[0]):
            for j in range(test_x.shape[1]):
                pred = 1 if y_pred_np[i][j] >= 0.5 else 0
                true = int(test_y_np[i][j])
                if pred == true:
                    correct_pos += 1

    pos_acc = correct_pos / total_pos
    print(f"\n测试集：总位置数{total_pos}，正确位置数{correct_pos}，位置准确率{pos_acc:.4f}")
    # 打印示例
    idx2char = {v: k for k, v in vocab.items()}
    sample_x = test_x[0].numpy()
    sample_y_true = test_y_np[0]
    sample_y_pred = y_pred_np[0]
    sample_chars = [idx2char[idx] for idx in sample_x]
    print("示例：")
    for char, true, pred_prob in zip(sample_chars, sample_y_true, sample_y_pred):
        pred = 1 if pred_prob >= 0.5 else 0
        print(f"字符：{char:2} | 真实标签：{true} | 预测概率：{pred_prob:.4f} | 预测结果：{pred}")
    return pos_acc

def main():
    vocab = build_vocab()
    model = RNNPositionModel(
        vocab_size=len(vocab),
        vector_dim=vector_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        batch_num = int(train_sample / batch_size)
        for _ in range(batch_num):
            train_x, train_y = build_dataset(batch_size, vocab)
            optim.zero_grad()
            loss = model(train_x, train_y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        pos_acc = evaluate(model, vocab)
        log.append([avg_loss, pos_acc])
        print(f"\n========= 第{epoch+1}轮 ==========")
        print(f"平均损失：{avg_loss:.6f} | 位置准确率：{pos_acc:.4f}")

    # 保存模型（增加weights_only=True参数）
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("\n模型和词表已保存！")

def predict(model_path, vocab_path, input_strings):
    # 加载词表和模型
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = RNNPositionModel(
        vocab_size=len(vocab),
        vector_dim=vector_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    # 修复警告：添加weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # 处理输入文本（关键修复：确保长度严格等于sentence_length）
    processed_x = []
    for s in input_strings:
        # 1. 截断过长文本
        truncated = s[:sentence_length]
        # 2. 计算需要补齐的长度
        pad_length = sentence_length - len(truncated)
        # 3. 生成索引序列（用vocab["pad"]补齐，而不是字符串"pad"）
        x = [vocab.get(char, vocab["unk"]) for char in truncated] + [vocab["pad"]] * pad_length
        processed_x.append(x)

    # 预测
    with torch.no_grad():
        x_tensor = torch.LongTensor(processed_x)  # 形状：[n, sentence_length]
        y_pred = model(x_tensor)
        y_pred_np = y_pred.squeeze(dim=2).numpy()

    # 输出结果
    print("\n预测结果：")
    idx2char = {v: k for k, v in vocab.items()}
    for i, s in enumerate(input_strings):
        print(f"\n输入文本：{s}")
        print(f"处理后文本：{''.join([idx2char[idx] for idx in processed_x[i]])}")
        print("位置详情：")
        for pos, (char_idx, pred_prob) in enumerate(zip(processed_x[i], y_pred_np[i])):
            char = idx2char[char_idx]
            pred = 1 if pred_prob >= 0.5 else 0
            is_target = "是特定字符" if pred == 1 else "非特定字符"
            print(f"位置{pos+1}：字符'{char}' | 预测概率：{pred_prob:.4f} | {is_target}")


if __name__ == "__main__":
    main()
    # 测试预测
    test_strings = ["你dasjda好", "我你dasa", "他在jwhuhqiwhiqwqw",]
    predict("model.pth", "vocab.json", test_strings)
