#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    """使用动态规划实现全切分"""
    n = len(sentence)
    if n == 0:
        return [[]]

    # 最大词长度
    max_word_len = max(len(word) for word in Dict) if Dict else 1

    # dp[i] 存储前i个字符的所有分词结果
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]  # 空字符串的分词结果

    for i in range(1, n + 1):
        # 尝试所有可能的词长
        start = max(0, i - max_word_len)
        for j in range(start, i):
            word = sentence[j:i]
            if word in Dict and dp[j]:
                # 将j位置的所有分词结果加上当前词
                for path in dp[j]:
                    new_path = path.copy()
                    new_path.append(word)
                    dp[i].append(new_path)

    target = dp[n]

    return target




#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

target_result = all_cut(sentence,Dict)
for i in target_result:
    print(i)
