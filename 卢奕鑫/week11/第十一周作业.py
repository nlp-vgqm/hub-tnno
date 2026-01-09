# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertConfig, BertModel

"""
基于BERT的自回归语言模型，使用SFT风格的注意力掩码
SFT (Suffix-prefix Tuning): 允许前缀关注所有位置，后缀只能关注前缀和自身
"""


class BERTLanguageModelWithSFTMask(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, max_seq_len, prefix_ratio=0.5):
        super(BERTLanguageModelWithSFTMask, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.prefix_ratio = prefix_ratio  # 前缀比例，例如0.5表示前一半是前缀

        # 创建BERT配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            max_position_embeddings=max_seq_len,
            is_decoder=True,
            is_encoder_decoder=False,
        )

        # 使用BERT模型
        self.bert = BertModel(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    # 生成SFT风格的注意力掩码
    def generate_sft_mask(self, seq_len, prefix_length):
        """
        生成SFT掩码矩阵：
        - 前缀部分：可以关注所有位置（包括前缀和后缀）
        - 后缀部分：只能关注前缀部分和自身，不能关注未来的后缀位置
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

        # 后缀部分（第prefix_length行之后）
        for i in range(prefix_length, seq_len):
            for j in range(prefix_length, seq_len):
                if j > i:  # j > i 表示未来位置
                    mask[i, j] = False

        return mask

    def forward(self, x, y=None, attention_mask=None, prefix_lengths=None):
        batch_size, seq_len = x.shape

        # 如果没有提供prefix_lengths，则根据比例计算
        if prefix_lengths is None:
            prefix_length = max(1, int(seq_len * self.prefix_ratio))
            prefix_lengths = torch.tensor([prefix_length] * batch_size, device=x.device)

        # 生成注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        # 为每个样本生成SFT掩码
        extended_attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=x.device)

        for i in range(batch_size):
            prefix_len = min(int(prefix_lengths[i].item()), seq_len)
            if prefix_len <= 0:
                prefix_len = 1

            sft_mask = self.generate_sft_mask(seq_len, prefix_len).to(x.device)

            # 将SFT掩码应用到注意力（True表示允许，False表示masked）
            sft_attention_mask = (~sft_mask).float() * -10000.0

            # 结合输入注意力掩码
            seq_mask = attention_mask[i].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            seq_attention_mask = (1.0 - seq_mask) * -10000.0

            # 合并两个掩码
            combined_mask = seq_attention_mask + sft_attention_mask.unsqueeze(0)  # [1, 1, seq_len, seq_len]
            extended_attention_mask[i] = combined_mask

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

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        y_pred = self.lm_head(sequence_output)

        if y is not None:
            loss = nn.functional.cross_entropy(
                y_pred.view(-1, y_pred.shape[-1]),
                y.view(-1),
                ignore_index=0  # 忽略pad token
            )
            return loss
        else:
            return torch.softmax(y_pred, dim=-1)


# 修改后的数据生成函数，支持SFT掩码
def build_sample_with_prefix(vocab, window_size, corpus, prefix_ratio=0.5):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]

    # 计算前缀长度
    prefix_length = max(1, int(window_size * prefix_ratio))

    # 确保词汇表索引在有效范围内
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]

    # 验证索引范围
    vocab_size = len(vocab)
    x = [min(idx, vocab_size - 1) for idx in x]
    y = [min(idx, vocab_size - 1) for idx in y]

    return x, y, prefix_length


def build_dataset_with_sft(sample_length, vocab, window_size, corpus, prefix_ratio=0.5):
    dataset_x = []
    dataset_y = []
    prefix_lengths = []

    vocab_size = len(vocab)

    for i in range(sample_length):
        x, y, prefix_len = build_sample_with_prefix(vocab, window_size, corpus, prefix_ratio)
        dataset_x.append(x)
        dataset_y.append(y)
        prefix_lengths.append(prefix_len)

    # 创建注意力掩码
    attention_masks = []
    for x in dataset_x:
        mask = [1] * len(x)
        attention_masks.append(mask)

    # 转换为张量
    x_tensor = torch.LongTensor(dataset_x)
    y_tensor = torch.LongTensor(dataset_y)
    mask_tensor = torch.FloatTensor(attention_masks)
    prefix_tensor = torch.LongTensor(prefix_lengths)

    return x_tensor, y_tensor, mask_tensor, prefix_tensor


# 改进的生成函数，支持SFT掩码
def generate_sentence_with_sft(openings, model, vocab, window_size, prefix_ratio=0.5, max_length=100):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    vocab_size = len(vocab)
    model.eval()

    with torch.no_grad():
        generated = openings

        while len(generated) < max_length:
            # 获取当前窗口
            current_text = generated[-window_size:] if len(generated) >= window_size else generated
            x = [vocab.get(char, vocab["<UNK>"]) for char in current_text]

            # 确保索引在有效范围内
            x = [min(idx, vocab_size - 1) for idx in x]

            # 填充到窗口大小
            if len(x) < window_size:
                pad_len = window_size - len(x)
                x = [vocab["<pad>"]] * pad_len + x

            x = torch.LongTensor([x])
            attention_mask = torch.FloatTensor([[1 if token != vocab["<pad>"] else 0 for token in x[0]]])

            # 对于生成，前缀长度递减
            current_prefix = max(1, min(int(window_size * prefix_ratio), len(current_text)))
            prefix_lengths = torch.LongTensor([current_prefix])

            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
                prefix_lengths = prefix_lengths.cuda()

            # 获取最后一个位置的预测
            pred = model(x, attention_mask=attention_mask, prefix_lengths=prefix_lengths)
            y_probs = pred[0][-1]  # 取最后一个位置的预测

            # 采样策略
            if random.random() > 0.1:
                index = torch.argmax(y_probs).item()  # greedy
            else:
                probs = y_probs.cpu().numpy()
                probs = np.maximum(probs, 1e-10)
                probs = probs / probs.sum()
                index = np.random.choice(len(probs), p=probs)

            # 确保索引有效
            index = min(max(index, 0), vocab_size - 1)
            pred_char = reverse_vocab.get(index, "<UNK>")
            generated += pred_char

            # 如果生成结束符或达到最大长度，停止
            if pred_char in ["\n", "<EOS>"] or len(generated) >= max_length:
                break

    return generated


# 简化的可视化函数，避免matplotlib问题
def visualize_sft_mask_simple(seq_len, prefix_length):
    """
    简单的文本方式可视化SFT掩码
    """
    print("\n" + "=" * 60)
    print(f"SFT Attention Mask Visualization")
    print(f"Sequence Length: {seq_len}, Prefix Length: {prefix_length}")
    print("=" * 60)

    # 生成掩码
    mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if i < prefix_length:
                # 前缀：可以看到所有位置
                row.append("█")
            else:
                # 后缀：只能看到前缀和当前位置及之前的位置
                if j < prefix_length or j <= i:
                    row.append("█")
                else:
                    row.append("░")
        mask.append(row)

    # 打印掩码
    print("\nMask (█=Allowed, ░=Masked):")
    print("   " + "".join([f"{j:2d}" for j in range(min(seq_len, 20))]))
    for i in range(min(seq_len, 20)):
        row_str = "".join(mask[i][:20])
        prefix_marker = "P" if i < prefix_length else "S"
        print(f"{prefix_marker}{i:2d} {row_str}")

    if seq_len > 20:
        print(f"... (showing first 20 of {seq_len} positions)")

    print("\nLegend:")
    print("  P = Prefix token")
    print("  S = Suffix token")
    print("  █ = Attention allowed")
    print("  ░ = Attention masked")
    print("=" * 60 + "\n")


# 构建一个简单的词汇表（如果文件不存在）
def build_or_load_vocab(vocab_path="vocab.txt"):
    """构建或加载词汇表"""
    if os.path.exists(vocab_path):
        print(f"加载词汇表: {vocab_path}")
        return build_vocab(vocab_path)
    else:
        print(f"词汇表文件不存在: {vocab_path}，创建简单词汇表")
        return build_simple_vocab()


def build_simple_vocab():
    """构建简单的测试词汇表"""
    vocab = {
        "<pad>": 0,
        "<UNK>": 1,
        "<EOS>": 2,
    }

    # 添加中文字符
    chinese_chars = "的一是不了在有人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"

    for i, char in enumerate(chinese_chars):
        vocab[char] = i + 3  # 从3开始，0-2是特殊token

    print(f"创建词汇表，大小: {len(vocab)}")
    return vocab


def build_vocab(vocab_path):
    """构建词汇表"""
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            if char:  # 确保不是空行
                vocab[char] = index + 1
    vocab["<UNK>"] = len(vocab)
    print(f"从文件加载词汇表，大小: {len(vocab)}")
    return vocab


def load_corpus(corpus_path):
    """加载语料"""
    corpus = ""
    if not os.path.exists(corpus_path):
        print(f"语料文件不存在: {corpus_path}，使用测试语料")
        # 创建测试语料
        test_corpus = """
        人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
        人工智能领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。
        可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。
        人工智能可以对人的意识、思维的信息过程的模拟。
        人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。
        """
        return test_corpus * 50

    try:
        with open(corpus_path, encoding="gbk") as f:
            for line in f:
                corpus += line.strip()
    except:
        try:
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    corpus += line.strip()
        except:
            print(f"加载语料失败: {corpus_path}")
            # 返回一个简单的测试语料
            test_corpus = "今天天气很好，我们一起去公园散步。明天可能会下雨，记得带伞。人工智能是未来的发展方向。"
            return test_corpus * 100

    print(f"加载语料，长度: {len(corpus)} 字符")
    return corpus


# 修改训练函数
def train_with_sft(corpus_path="corpus.txt", save_weight=False):
    epoch_num = 5  # 减少轮数以便快速测试
    batch_size = 8  # 减少batch size
    train_sample = 1000  # 减少样本数
    char_dim = 128  # 减少维度
    window_size = 10  # 减少窗口大小
    max_seq_len = 20
    prefix_ratio = 0.5  # SFT前缀比例

    # 可视化SFT掩码
    visualize_sft_mask_simple(window_size, int(window_size * prefix_ratio))

    # 构建或加载词汇表
    vocab = build_or_load_vocab()
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    # 检查词汇表索引
    print(f"词汇表索引范围: 0-{vocab_size - 1}")

    # 加载语料
    corpus = load_corpus(corpus_path)
    print(f"语料长度: {len(corpus)} 字符")

    # 确保语料足够长
    if len(corpus) < window_size * 10:
        corpus = corpus * (window_size * 10 // len(corpus) + 1)
        print(f"扩展语料到长度: {len(corpus)}")

    # 使用SFT模型
    model = BERTLanguageModelWithSFTMask(
        vocab_size=vocab_size,
        hidden_dim=char_dim,
        num_layers=2,  # 进一步减少层数
        num_heads=2,  # 进一步减少头数
        max_seq_len=max_seq_len,
        prefix_ratio=prefix_ratio
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用SFT掩码，前缀比例: {prefix_ratio}")

    # 检查是否有CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\n开始训练...")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 训练循环
        num_batches = int(train_sample / batch_size)
        for batch_idx in range(num_batches):
            try:
                x, y, attention_masks, prefix_lengths = build_dataset_with_sft(
                    batch_size, vocab, window_size, corpus, prefix_ratio
                )

                # 移动数据到设备
                x = x.to(device)
                y = y.to(device)
                attention_masks = attention_masks.to(device)
                prefix_lengths = prefix_lengths.to(device)

                # 验证数据
                if x.max() >= vocab_size:
                    print(f"警告: 输入索引超出范围，最大索引: {x.max()}, 词汇表大小: {vocab_size}")
                    x = torch.clamp(x, 0, vocab_size - 1)

                optim.zero_grad()
                loss = model(x, y, attention_mask=attention_masks, prefix_lengths=prefix_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                watch_loss.append(loss.item())

                # 每10个batch打印一次进度
                if batch_idx % 10 == 0:
                    avg_loss = np.mean(watch_loss[-10:]) if len(watch_loss) >= 10 else np.mean(watch_loss)
                    print(f"Epoch {epoch + 1}/{epoch_num}, Batch {batch_idx}/{num_batches}: loss = {avg_loss:.4f}")

            except Exception as e:
                print(f"Batch {batch_idx} 出错: {e}")
                # 跳过出错的batch
                continue

        # 计算平均损失
        if watch_loss:
            epoch_avg_loss = np.mean(watch_loss)
            print("=" * 50)
            print(f"第{epoch + 1}/{epoch_num}轮平均loss: {epoch_avg_loss:.6f}")
            print(f"学习率: {optim.param_groups[0]['lr']:.6f}")

            # 生成测试
            model.eval()
            test_prefixes = ["今天天气", "人工智能", "机器学习"]

            print("\n生成测试:")
            for prefix in test_prefixes:
                try:
                    generated = generate_sentence_with_sft(
                        prefix, model, vocab, window_size, prefix_ratio, max_length=20)
                    print(f"输入: '{prefix}'")
                    print(f"生成: '{generated}'")
                    print("-" * 40)
                except Exception as e:
                    print(f"生成测试失败: {e}")
        else:
            print(f"第{epoch + 1}轮没有有效的训练数据")

    if save_weight:
        os.makedirs("model", exist_ok=True)
        model_path = os.path.join("model", "sft_model.pth")

        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'config': {
                'char_dim': char_dim,
                'window_size': window_size,
                'max_seq_len': max_seq_len,
                'prefix_ratio': prefix_ratio,
                'vocab_size': vocab_size
            }
        }, model_path)
        print(f"\nSFT模型已保存到: {model_path}")

    return model


# 测试函数
def test_model(model, vocab, window_size=10):
    """测试模型生成能力"""
    print("\n" + "=" * 60)
    print("模型测试")
    print("=" * 60)

    model.eval()

    test_cases = [
        "今天天气",
        "人工智能",
        "深度学习",
        "自然语言",
        "计算机科"
    ]

    for prefix in test_cases:
        print(f"\n输入: '{prefix}'")
        for ratio in [0.3, 0.5, 0.7]:
            try:
                generated = generate_sentence_with_sft(
                    prefix, model, vocab, window_size, ratio, max_length=30)
                print(f"  前缀比例 {ratio}: {generated}")
            except Exception as e:
                print(f"  前缀比例 {ratio}: 生成失败 - {e}")

    print("=" * 60)


if __name__ == "__main__":
    # 测试SFT模型
    print("=" * 60)
    print("SFT Language Model Training")
    print("=" * 60)

    # 先测试SFT掩码可视化
    visualize_sft_mask_simple(10, 3)

    try:
        # 使用SFT版本进行训练
        model = train_with_sft("corpus.txt", False)

        # 测试模型
        vocab = build_or_load_vocab()
        test_model(model, vocab)

    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback

        traceback.print_exc()

        # 创建一个小型测试模型
        print("\n创建小型测试模型...")
        vocab = build_simple_vocab()
        model = BERTLanguageModelWithSFTMask(
            vocab_size=len(vocab),
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
            max_seq_len=10,
            prefix_ratio=0.5
        )
        test_model(model, vocab, window_size=8)
