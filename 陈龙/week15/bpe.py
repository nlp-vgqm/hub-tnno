import regex as re

# 1. 准备语料 读取 三体.txt
with open("三体.txt", "r", encoding="GBK") as f:
    text = f.read()

# 将文本转为 UTF-8 字节流，并转为整数列表 (0-255)
tokens = list(text.encode("utf-8"))

def get_stats(ids):
    """统计当前序列中所有相邻字符对的出现频率"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """将序列中的指定字符对(pair)替换为新的token id(idx)"""
    newids = []
    i = 0
    while i < len(ids):
        # 如果不是最后一个字符，且当前字符与下一个字符匹配 pair
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# -----------------------------------------------------------------------------
# 训练 BPE (生成词表)
# -----------------------------------------------------------------------------

# 设定目标词表大小
vocab_size = 4096
num_merges = vocab_size - 256

# 备份一份 tokens 用于训练
ids = list(tokens)

merges = {}  # 记录合并规则: (int, int) -> int

print(f"开始训练 BPE，目标词表大小: {vocab_size}...")
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        print("没有更多可合并的字符对，提前结束。")
        break

    # 找到出现次数最多的 pair
    pair = max(stats, key=stats.get)

    # 如果出现次数只有1次，对于大语料可能就不合并了，但这里我们继续
    if stats[pair] < 2:
        # 实际训练中通常会设置一个阈值，比如出现频率低于10就不合并了
        pass

    idx = 256 + i
    # print(f"合并: {pair} -> 新Token: {idx} (出现次数: {stats[pair]})")

    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("训练完成！")
print(f"原始字节长度: {len(tokens)}")
print(f"压缩后Token长度: {len(ids)}")
print(f"压缩比: {len(tokens) / len(ids):.2f}X")

# -----------------------------------------------------------------------------
# 构建最终词表 (用于解码)
# -----------------------------------------------------------------------------
# 初始词表：0-255 对应其字节本身
vocab = {idx: bytes([idx]) for idx in range(256)}

# 根据 merge 规则，由底向上构建词表
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


# -----------------------------------------------------------------------------
# 对外提供的函数 (Encoder & Decoder)
# -----------------------------------------------------------------------------

def encode_text(text):
    """
    将句子转换为序列 (Encoder)
    逻辑：先转为 byte 数组，然后按照训练好的 merges 规则，按优先级顺序进行合并
    """
    tokens = list(text.encode("utf-8"))

    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 这一步很关键：我们需要找到当前 tokens 里存在的，且在 merges 表中出现最早(优先级最高)的 pair
        # 使用 min 配合 key 查找 merges 中 value (即 idx) 最小的那个 pair
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        if pair not in merges:
            break  # 当前序列中没有任何可以合并的规则了

        idx = merges[pair]
        tokens = merge(tokens, pair, idx)

    return tokens


def decode_ids(ids):
    """
    将序列转换为句子 (Decoder)
    逻辑：将 token id 映射回 bytes，拼接后尝试用 utf-8 解码
    """
    # 将所有 token ID 转换回对应的 byte 串
    byte_list = b"".join(vocab[idx] for idx in ids)
    # 解码为字符串，errors="replace" 用于处理可能出现的解码错误（如截断的utf-8字符）
    text = byte_list.decode("utf-8", errors="replace")
    return text


# -----------------------------------------------------------------------------
# 测试验证
# -----------------------------------------------------------------------------
print("\n--- 测试 ---")
test_str = "汪教授，你了解科学边界吗？"
print(f"原文: {test_str}")

# 编码
encoded_ids = encode_text(test_str)
print(f"Encoded IDs: {encoded_ids}")

# 解码
decoded_str = decode_ids(encoded_ids)
print(f"Decoded Text: {decoded_str}")

# 验证一致性
assert test_str == decoded_str
print("验证通过：解码后内容与原文一致。")

# 查看生成的词表中有趣的内容 (如果是中文语料)
# 我们打印最后生成的5个 token 看看它学到了什么
print("\n--- 词表末尾展示 (学到的高层语义) ---")
for i in range(max(0, len(merges) - 5), len(merges)):
    idx = 256 + i
    # 反向查找 pair
    pair = [k for k, v in merges.items() if v == idx][0]
    token_bytes = vocab[idx]
    try:
        token_str = token_bytes.decode("utf-8")
        print(f"Token {idx}: {token_str} (由 {pair} 合并)")
    except:
        print(f"Token {idx}: {token_bytes} (部分字节，无法显示)")
