def bpe_tokenize(corpus, num_merges=4):
    # 步骤1：初始化基础字符集（去重+排序）
    chars = sorted(list(set([c for c in corpus if c != ' '])))
    vocab = {c: i for i, c in enumerate(chars)}  # 字符→索引映射
    print(f"初始词表：{list(vocab.keys())}\n")

    # 步骤2：语料预处理（按空格分割为词，每个词拆分为字符列表）
    corpus_words = [list(word) for word in corpus.split()]

    # 步骤3：迭代合并高频字符对
    for merge_idx in range(num_merges):
        # 统计相邻字符对频率
        pair_freq = {}
        for word in corpus_words:
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                pair_freq[pair] = pair_freq.get(pair, 0) + 1

        # 选择最高频字符对
        if not pair_freq:
            break
        best_pair = max(pair_freq.items(), key=lambda x: x[1])[0]
        print(f"第{merge_idx+1}轮合并：{best_pair}（频率：{pair_freq[best_pair]}）")

        # 合并字符对（生成新token）
        new_token = ''.join(best_pair)
        new_vocab_idx = len(vocab)
        vocab[new_token] = new_vocab_idx

        # 更新语料（将所有best_pair替换为new_token）
        new_corpus_words = []
        for word in corpus_words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus_words.append(new_word)
        corpus_words = new_corpus_words
        print(f"更新后词表：{list(vocab.keys())}\n")

    # 步骤4：最终分词（展平所有词的token列表）
    final_tokens = [token for word in corpus_words for token in word]
    return final_tokens, vocab

# 运行BPE分词
corpus = "i love machine learning machine learning is fun i love coding too coding makes me happy"
tokens, vocab = bpe_tokenize(corpus, num_merges=4)
print(f"最终分词结果：{tokens}")
print(f"词表规模：{len(vocab)}")
