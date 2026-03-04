import re


def bpe_train(text):
    # 直接查找所有连续的中文、字母、数字
    pattern = re.compile(r'[\u4e00-\u9fffa-zA-Z0-9]+')
    datas = pattern.findall(text)
    vocab = {}
    #统计文本中共有多少个不同的中文、字母、数字，并建立词表
    for data in datas:
        for i in range(len(data)):
            if data[i] not in vocab:
                vocab[data[i]] = len(vocab) + 1
    newid = {}  # 用于存储发现的连续字符序列

    for data in datas:
        if len(data) < 2:
            continue

        n = len(data)

        # 从长度为2开始，逐步增加长度
        current_length = 2

        while current_length <= n:
            # 统计当前长度下所有连续字符序列的出现次数
            sequence_counts = {}

            # 遍历所有可能的连续序列
            for i in range(n - current_length + 1):
                seq = data[i:i + current_length]
                if seq in sequence_counts:
                    sequence_counts[seq] += 1
                else:
                    sequence_counts[seq] = 1

            # 找出出现次数大于1的序列
            new_sequences = {}
            for seq, count in sequence_counts.items():
                if count > 1 and seq not in newid and seq not in vocab:
                    # 添加到newid中，值为原newid中总键数加1
                    newid[seq] = len(newid) + 1
                    new_sequences[seq] = count

            # 如果有新的序列被发现，继续增加长度
            if new_sequences:
                current_length += 1
            else:
                # 如果没有发现新序列，检查下一个长度
                current_length += 1

                # 如果长度已经超过数据长度的一半，可以提前终止
                if current_length > n // 2 + 1:
                    break

    print("\n发现的连续序列 (newid):", newid)

    # 3. 将所有新发现的序列也加入vocab
    for seq in newid:
        if seq not in vocab:
            vocab[seq] = len(vocab) + 1

    print("\n最终词汇表 (vocab):", vocab)

    return vocab, newid


if __name__ == '__main__':
    with open('corpus.txt', 'r', encoding='gbk') as f:
        text = f.read()
    # text = '一二三 abc!@#ABC<,123.aa'
    vocab, newid = bpe_train(text)
    text_input = input('请输入需要查询的词')
    if text_input in vocab:
        print(vocab[text_input])
    else:
        print('你查询的词不存在')
