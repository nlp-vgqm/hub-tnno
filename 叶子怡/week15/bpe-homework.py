from collections import defaultdict

def get_top_pairs(tokens: list):
    counts = {}
    for pairs in zip(tokens, tokens[1:]):
        counts[pairs] = counts.get(pairs, 0) + 1
    # sorted_count = sorted(((v, k) for k, v in counts.items()), reverse=True)
    top_pair = max(counts, key=counts.get)
    return top_pair


def merge(tokens: list, top_pair: tuple, idx: int):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i+1 < len(tokens) and (tokens[i], tokens[i+1]) == top_pair:
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


# bpe编码
def bpe_encode(tokens: list, vocab_size=276):
    idx = 256
    i = 0
    idx_pair = {} # bpe编码词典
    while i <  vocab_size - idx:
        new_idx = idx + i
        top_pair = get_top_pairs(tokens)

        idx_pair[new_idx] = top_pair
        print("merging ", top_pair, " into a new token ", new_idx)

        tokens = merge(tokens, top_pair, new_idx)
        i += 1

    return tokens, idx_pair

def separate(tokens: list, idx_pair: dict, idx: int):
    i = 0
    dec_tokens = []
    while i < len(tokens):
        if tokens[i] == idx:
            dec_tokens.append(idx_pair[idx][0])
            dec_tokens.append(idx_pair[idx][1])
        else:
            dec_tokens.append(tokens[i])
        i += 1
    return dec_tokens

# 解码
def bpe_decode(tokens: list, idx_pair: dict, vocab_size=276):
    idx = vocab_size - 1
    while idx >= 256:
        tokens = separate(tokens, idx_pair, idx)
        print("decode ", idx, "to ", idx_pair[idx])
        idx -= 1
    return tokens

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


text = load_corpus(r"corpus.txt")
utf8_tokens = list(text.encode("utf-8"))
print(utf8_tokens)
tokens_len = len(utf8_tokens)
print("before bpe len:", tokens_len)

# bpe编码
encode_tokens, enc_vocab = bpe_encode(utf8_tokens)
print("after bpe len", len(encode_tokens))

# bpe解码
decode_tokens = bpe_decode(encode_tokens, enc_vocab)
if utf8_tokens == decode_tokens:
    print("解码正确！")
else:
    print("解码错误！解码后长度：", len(decode_tokens))


