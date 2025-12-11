# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import os

"""
优化版：提升中文文本的目标字符定位准确率
1. 扩充中文字符集，减少未知字符比例
2. 增加中文样本占比，增强模型对中文环境的适应
3. 优化损失函数权重，减少对填充/未知字符的误判
"""


class PositionLocator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(PositionLocator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2  # 增加LSTM层数，增强特征提取能力
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        # 增加正样本权重（解决类别不平衡，尤其中文样本中目标字符占比低的问题）
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))  # 正样本权重设为3

    def forward(self, x, y=None):
        x_embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x_embed)  # (batch, seq_len, hidden*2)
        logits = self.fc(lstm_out).squeeze(-1)  # (batch, seq_len)

        if y is not None:
            # 忽略填充符位置的损失（填充符标签为0，但不应参与损失计算）
            mask = (x != 0).float()  # 填充符（索引0）的mask为0，其他为1
            loss = self.loss_fn(logits * mask, y * mask)  # 只计算有效字符的损失
            return loss
        else:
            return self.sigmoid(logits)


def build_vocab(target_char='国'):
    """扩充中文字符集，包含常见汉字（解决全中文文本被识别为<unk>的问题）"""
    # 核心：增加500个常见中文字符（覆盖日常用语）
    common_chars = """的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"""
    # 基础字符集 = 目标字符 + 常见汉字 + 其他字符（字母、符号等）
    base_chars = target_char + common_chars + "abcdefghijklmnopqrstuvwxyz0123456789，。！？、；：“”‘’（）【】 "
    vocab = {"_": 0}  # 填充符
    # 去重并构建词表
    unique_chars = list(set(base_chars))  # 去重避免重复
    for idx, char in enumerate(unique_chars):
        vocab[char] = idx + 1
    vocab["<unk>"] = len(vocab)  # 未知字符
    return vocab


def generate_sample(vocab, seq_len, target_char='国'):
    """优化样本生成：增加中文样本比例，确保目标字符在中文语境中出现"""
    # 分离中文字符和非中文字符（便于控制中文样本比例）
    chinese_chars = [c for c in vocab.keys() if '\u4e00' <= c <= '\u9fff' and c not in ["_", "<unk>"]]
    other_chars = [c for c in vocab.keys() if not ('\u4e00' <= c <= '\u9fff') and c not in ["_", "<unk>"]]

    # 70%概率生成中文主导的样本，30%生成混合样本（提升中文适应性）
    if random.random() < 0.7:
        # 中文样本：主要从中文字符中选择
        valid_chars = chinese_chars + [target_char] * 3  # 提高目标字符在中文样本中的出现率
    else:
        # 混合样本：中英文混合
        valid_chars = chinese_chars + other_chars + [target_char] * 2

    # 生成序列
    sequence = [random.choice(valid_chars) for _ in range(seq_len)]
    # 生成标签
    labels = [1.0 if char == target_char else 0.0 for char in sequence]
    # 转换为索引
    seq_indices = [vocab[char] for char in sequence]
    return seq_indices, labels


def build_dataset(sample_num, vocab, seq_len, target_char='国'):
    X, Y = [], []
    for _ in range(sample_num):
        x, y = generate_sample(vocab, seq_len, target_char)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.FloatTensor(Y)


def evaluate_model(model, vocab, seq_len, target_char='国'):
    """增强评估：专门增加全中文测试样本，针对性评估"""
    model.eval()
    # 生成200个混合样本 + 100个全中文样本
    x_mix, y_mix = build_dataset(200, vocab, seq_len, target_char)
    # 生成全中文测试样本
    chinese_chars = [c for c in vocab.keys() if '\u4e00' <= c <= '\u9fff' and c not in ["_", "<unk>"]]
    x_zh, y_zh = [], []
    for _ in range(100):
        seq = [random.choice(chinese_chars + [target_char] * 2) for _ in range(seq_len)]
        y = [1.0 if c == target_char else 0.0 for c in seq]
        x = [vocab[c] for c in seq]
        x_zh.append(x)
        y_zh.append(y)
    x_zh = torch.LongTensor(x_zh)
    y_zh = torch.FloatTensor(y_zh)

    # 合并评估
    with torch.no_grad():
        # 混合样本评估
        y_pred_mix = model(x_mix)
        y_pred_label_mix = (y_pred_mix >= 0.5).float()
        # 全中文样本评估
        y_pred_zh = model(x_zh)
        y_pred_label_zh = (y_pred_zh >= 0.5).float()

    # 计算混合样本准确率
    total_mix = y_mix.numel()
    correct_mix = (y_pred_label_mix == y_mix).sum().item()
    acc_mix = correct_mix / total_mix
    # 计算全中文样本准确率（重点关注）
    total_zh = y_zh.numel()
    correct_zh = (y_pred_label_zh == y_zh).sum().item()
    acc_zh = correct_zh / total_zh

    # 目标字符专项评估
    target_total_mix = y_mix.sum().item()
    target_correct_mix = ((y_pred_label_mix == 1) & (y_mix == 1)).sum().item() if target_total_mix > 0 else 0
    target_acc_mix = target_correct_mix / target_total_mix if target_total_mix > 0 else 0

    target_total_zh = y_zh.sum().item()
    target_correct_zh = ((y_pred_label_zh == 1) & (y_zh == 1)).sum().item() if target_total_zh > 0 else 0
    target_acc_zh = target_correct_zh / target_total_zh if target_total_zh > 0 else 0

    print(f"\n评估结果：")
    print(f"混合样本准确率：{correct_mix}/{total_mix} = {acc_mix:.4f}")
    print(f"全中文样本准确率：{correct_zh}/{total_zh} = {acc_zh:.4f} （重点）")
    print(f"混合样本目标字符准确率：{target_correct_mix}/{target_total_mix} = {target_acc_mix:.4f}")
    print(f"全中文样本目标字符准确率：{target_correct_zh}/{target_total_zh} = {target_acc_zh:.4f} （重点）")
    return acc_zh  # 以全中文准确率为主要参考


def train(target_char='国'):
    # 超参数调整：增加训练样本和轮数，提升中文学习效果
    seq_len = 20  # 适当增加序列长度，适应更长中文文本
    embed_dim = 64  # 增加嵌入维度，更好区分中文字符
    hidden_dim = 128  # 增加LSTM隐藏层维度
    epoch_num = 30  # 增加训练轮数
    batch_size = 32
    train_samples = 30000  # 增加训练样本量（尤其是中文样本）
    lr = 0.001

    vocab = build_vocab(target_char)
    vocab_size = len(vocab)
    print(f"词表大小：{vocab_size}（包含{sum(1 for c in vocab if '\u4e00' <= c <= '\u9fff')}个中文字符）")

    model = PositionLocator(vocab_size, embed_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # 学习率衰减

    best_zh_acc = 0  # 跟踪全中文样本的最佳准确率
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0.0
        batch_count = train_samples // batch_size

        for _ in range(batch_count):
            x_batch, y_batch = build_dataset(batch_size, vocab, seq_len, target_char)
            optimizer.zero_grad()
            loss = model(x_batch, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / batch_count
        print(f"\n第{epoch + 1}轮训练")
        print(f"平均损失：{avg_loss:.6f}")
        current_zh_acc = evaluate_model(model, vocab, seq_len, target_char)

        # 保存全中文准确率最高的模型
        if current_zh_acc > best_zh_acc:
            best_zh_acc = current_zh_acc
            torch.save(model.state_dict(), "best_position_locator.pth")
            print(f"保存最优模型（全中文准确率：{best_zh_acc:.4f}）")

        scheduler.step(avg_loss)  # 根据损失调整学习率

    # 保存最终配置
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open("config.json", "w") as f:
        json.dump({
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "target_char": target_char,
            "pad_char": "_"
        }, f)
    print("\n训练完成，最优模型保存为best_position_locator.pth")


def predict_positions(input_texts):
    with open("config.json", "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]
    hidden_dim = config["hidden_dim"]
    target_char = config["target_char"]
    pad_char = config["pad_char"]

    with open("vocab.json", "r", encoding="utf8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    model = PositionLocator(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load("best_position_locator.pth"))  # 加载最优模型
    model.eval()

    processed_texts = []
    input_indices = []
    for text in input_texts:
        # 处理输入文本
        if len(text) > seq_len:
            processed = text[:seq_len]
        else:
            processed = text.ljust(seq_len, pad_char)
        # 转换为索引（打印未知字符，便于调试）
        indices = []
        for char in processed:
            if char not in vocab:
                print(f"警告：字符'{char}'未在词表中，将被视为<unk>")
            indices.append(vocab.get(char, vocab["<unk>"]))
        processed_texts.append(processed)
        input_indices.append(indices)

    with torch.no_grad():
        x_tensor = torch.LongTensor(input_indices)
        pred_probs = model(x_tensor)
        pred_labels = (pred_probs >= 0.5).float()

    # 输出结果
    for i, text in enumerate(input_texts):
        print(f"\n输入文本：{text}")
        print(f"目标字符：'{target_char}'")
        positions = [idx for idx, label in enumerate(pred_labels[i]) if label == 1.0]

        if positions:
            print(f"检测到位置：{positions}")
            print("对应字符及概率：")
            for p in positions:
                print(f"位置{p}：'{processed_texts[i][p]}'，概率：{pred_probs[i][p]:.4f}")
        else:
            print("未检测到目标字符")


if __name__ == "__main__":
    train(target_char='国')  # 可替换为其他中文字符，如'train(target_char="中")'

    # 重点测试全中文文本
    test_texts = [
        "中国人民共和国",
        "我爱我的祖国",
        "国旗下的讲话内容",
        "国庆节快乐",
        "这个文本里没有目标字",
        "国国国国国",
        "山川河流日月星辰",  # 无目标字符的全中文文本
        "国中园囿国国国",
        "AAS国DF"
    ]
    predict_positions(test_texts)
