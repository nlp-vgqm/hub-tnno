# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import os

"""
基于pytorch的RNN模型，判断字符"a"在文本中的位置（多分类任务）
- 统一配置管理，确保训练/预测参数一致
- 修复输入序列长度不一致问题
"""

# ===================== 全局配置（统一训练和预测的参数）=====================
CONFIG = {
    "epoch_num": 10,
    "batch_size": 40,
    "train_sample": 1000,
    "char_dim": 30,  # 字符嵌入维度（统一）
    "sentence_length": 20,  # 输入序列固定长度（训练/预测必须一致！）
    "learning_rate": 0.001,
    "model_path": "divNlpDemo.pth",
    "vocab_path": "vocab.json",
    "config_path": "config.json"  # 保存配置，方便预测时加载
}


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # 需Long输入
        self.rnnlayer = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 输出维度：位置数+1（不存在的情况）
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len) -> (batch_size, seq_len, vector_dim)
        output, hidden = self.rnnlayer(x)
        x = output[:, -1, :]
        # x = hidden.squeeze()  # 取RNN最后一个时刻输出
        y_pred = self.classify(x)  # (batch_size, sentence_length+1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}  # pad对应索引0，用于填充
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)  # 未知字符索引
    return vocab


def build_sample(vocab, sentence_length):
    """生成单个样本，确保x长度为sentence_length"""
    # 随机采样，确保长度为sentence_length（允许重复采样）
    x_chars = random.choices(list(vocab.keys()), k=sentence_length)
    # 确定标签y
    y = x_chars.index('a') if 'a' in x_chars else sentence_length
    # 转换为索引序列
    x = [vocab.get(char, vocab['unk']) for char in x_chars]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    """生成数据集，返回LongTensor（确保序列长度一致）"""
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab):
    """根据全局配置构建模型"""
    model = TorchModel(
        vector_dim=CONFIG["char_dim"],
        sentence_length=CONFIG["sentence_length"],
        vocab=vocab
    )
    return model


def evaluate(model, vocab):
    """评估模型，使用全局配置的参数"""
    model.eval()
    x, y = build_dataset(
        sample_length=200,
        vocab=vocab,
        sentence_length=CONFIG["sentence_length"]
    )
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0.0
    print(f"正确预测个数：{correct}, 错误个数：{wrong}, 正确率：{accuracy:.4f}")
    return accuracy


def save_config():
    """保存配置到文件，方便预测时加载"""
    with open(CONFIG["config_path"], "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)


def load_config():
    """预测时加载训练时的配置，确保参数一致"""
    if os.path.exists(CONFIG["config_path"]):
        with open(CONFIG["config_path"], "r", encoding="utf8") as f:
            loaded_config = json.load(f)
        CONFIG.update(loaded_config)  # 更新全局配置
    else:
        raise FileNotFoundError("配置文件不存在，请先训练模型！")


def main():
    # 保存配置（供预测时使用）
    save_config()
    # 构建词表和模型
    vocab = build_vocab()
    model = build_model(vocab)
    optim = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    log = []

    # 训练循环
    for epoch in range(CONFIG["epoch_num"]):
        model.train()
        watch_loss = []
        batch_num = int(CONFIG["train_sample"] / CONFIG["batch_size"])
        for _ in range(batch_num):
            x, y = build_dataset(
                sample_length=CONFIG["batch_size"],
                vocab=vocab,
                sentence_length=CONFIG["sentence_length"]
            )
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        # 打印训练信息
        print(f"\n========= 第{epoch + 1}轮 =========")
        print(f"平均loss: {np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型和词表
    torch.save(model.state_dict(), CONFIG["model_path"])
    with open(CONFIG["vocab_path"], "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("\n模型、词表和配置已保存")


def predict(input_strings):
    """预测函数：加载训练时的配置，确保输入长度一致"""
    # 加载训练时的配置（关键！避免参数不一致）
    load_config()
    # 加载词表和模型
    if not os.path.exists(CONFIG["vocab_path"]) or not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError("词表或模型文件不存在，请先训练！")
    vocab = json.load(open(CONFIG["vocab_path"], "r", encoding="utf8"))
    model = build_model(vocab)
    model.load_state_dict(torch.load(CONFIG["model_path"], weights_only=True))
    model.eval()

    # ===================== 关键：统一输入序列长度 =====================
    x = []
    for s in input_strings:
        # 1. 转换为索引序列（未知字符用unk）
        idx_list = [vocab.get(char, vocab['unk']) for char in s]
        # 2. 截断：超长部分舍弃（保留前sentence_length个）
        if len(idx_list) > CONFIG["sentence_length"]:
            idx_list = idx_list[:CONFIG["sentence_length"]]
        # 3. 补0：不足部分用pad的索引（0）填充
        if len(idx_list) < CONFIG["sentence_length"]:
            idx_list += [vocab["pad"]] * (CONFIG["sentence_length"] - len(idx_list))
        # 验证长度（确保一致）
        assert len(idx_list) == CONFIG["sentence_length"], f"输入处理后长度为{len(idx_list)}，期望{CONFIG['sentence_length']}"
        x.append(idx_list)

    # 预测
    with torch.no_grad():
        x_tensor = torch.LongTensor(x)  # 所有子列表长度一致，可正常构建张量
        y_pred = model(x_tensor)
        y_prob = torch.softmax(y_pred, dim=1)  # 转换为置信度

    # 输出结果
    print("\n======= 预测结果 =======")
    for s, pred, prob in zip(input_strings, y_pred, y_prob):
        pred_label = torch.argmax(pred).item()
        # 解析结果
        if pred_label == CONFIG["sentence_length"]:
            result = "不存在字符'a'"
        else:
            # 注意：输入可能被截断，需提示用户
            original_has_a = 'a' in s
            if original_has_a and pred_label < len(s):
                result = f"字符'a'的位置：{pred_label}（原始输入）"
            else:
                result = f"字符'a'的位置：{pred_label}（输入已截断/补全）"
        max_prob = prob[pred_label].item()
        print(f"输入：{s:20s} | 预测结果：{result:30s} | 置信度：{max_prob:.4f}")


if __name__ == "__main__":
    # 训练模型（首次运行时执行）
    main()
    # 测试预测（训练完成后执行）
    test_strings = [
        "fnvfa", "wza", "aqwdeg", "akwww", "xyz",
        "abcdefghijklmnopqrst",  # 长度20（等于sentence_length）
        "abcdefghijklmnopqrstuvwxyz"  # 长度26（超长，会截断）
    ]
    predict(test_strings)