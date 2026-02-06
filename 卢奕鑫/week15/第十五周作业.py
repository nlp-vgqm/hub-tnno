import json
import re
from collections import defaultdict


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, vocab):
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
    for word, freq in vocab.items():
        w_out = pattern.sub(replacement, word)
        v_out[w_out] = freq
    return v_out


def train_bpe(text, num_merges=100):
    # 初始化词汇表：将文本拆分为字符级
    vocab = defaultdict(int)
    words = re.findall(r'\S+|\s', text)  # 保留空格作为单独token
    for word in words:
        vocab[' '.join(list(word))] += 1

    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)

    return vocab, merges


def apply_bpe(text, merges):
    for pair in merges:
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        text = text.replace(bigram, replacement)
    return text


# 1. 读取tag_news.json文件
with open('tag_news.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
print("\n读取tag_news.json文件")
# 2. 提取所有文本内容（标题+内容）
corpus = ""
for item in data:
    corpus += item.get('title', '') + " "
    corpus += item.get('content', '') + " "
print("\n提取所有文本内容（标题+内容）")
# 3. 训练BPE
vocab, merges = train_bpe(corpus, num_merges=500)
print("\n训练BPE")
# 4. 查看部分合并结果
print("Top 10 merges:")
for i, merge in enumerate(merges[:10]):
    print(f"{i + 1}: {merge[0]} + {merge[1]} -> {''.join(merge)}")

# 5. 测试分词
test_text = "新增资金入场沪胶强势创年内新高"
tokenized = apply_bpe(test_text, merges)
print("\n测试文本分词结果:")
print(tokenized)

# 6. 保存词表
with open('bpe_vocab.txt', 'w', encoding='utf-8') as f:
    for word, freq in vocab.items():
        f.write(f"{word}\t{freq}\n")

print("\nBPE训练完成，词表已保存为 bpe_vocab.txt")
