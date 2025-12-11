def all_cut(sentence, word_dict):
    """
    返回句子按照字典切分的所有可能结果

    Args:
        sentence: 待切分的字符串
        word_dict: 词典，包含词语及其权重

    Returns:
        list: 包含所有可能切分方式的双层列表
    """

    def backtrack(start, path, result):
        # 如果已经处理完整个句子，将当前路径添加到结果中
        if start == len(sentence):
            result.append(path[:])  # 使用切片创建副本
            return

        # 尝试所有可能的切分
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果当前子串在词典中，继续递归
            if word in word_dict:
                path.append(word)
                backtrack(end, path, result)
                path.pop()  # 回溯

    result = []
    backtrack(0, [], result)
    return result


# 测试数据
Dict = {"经常": 0.1, "经": 0.05, "有": 0.1, "常": 0.001, "有意见": 0.1,
        "歧": 0.001, "意见": 0.2, "分歧": 0.2, "见": 0.05, "意": 0.05,
        "见分歧": 0.05, "分": 0.1}

sentence = "经常有意见分歧"

# 调用函数
results = all_cut(sentence, Dict)

# 输出结果
print("所有可能的切分结果：")
for i, result in enumerate(results, 1):
    print(f"{i}: {result}")







