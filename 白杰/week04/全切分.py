# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    result = []  # 存储所有切分结果
    n = len(sentence)  # 句子长度

    # 回溯函数：递归尝试所有切分可能
    # index：当前处理到句子的第几个字符（起始位置）
    # path：当前已切分的词列表
    def backtrack(index, path):
        # 终止条件：处理完整个句子（起始位置到达句子末尾）
        if index == n:
            result.append(path.copy())  # 复制当前路径，避免后续修改影响结果
            return
        
        # 遍历所有可能的结束位置（从index+1到句子末尾）
        for end in range(index + 1, n + 1):
            # 截取子串：从index到end（左闭右开）
            current_word = sentence[index:end]
            # 如果子串在字典中（是有效词），则继续递归切分剩余部分
            if current_word in Dict:
                path.append(current_word)  # 把当前有效词加入路径
                backtrack(end, path)       # 递归处理end之后的子串
                path.pop()                 # 回溯：移除当前词，尝试下一种可能

    # 从句子第0个字符开始，初始路径为空
    backtrack(index=0, path=[])
    return result

# 测试函数
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

# 验证结果（不考虑顺序，对比集合形式）
result = all_cut(sentence, Dict)
# 转换为不可变类型（tuple）才能比较集合
result_set = set(tuple(seq) for seq in result)
target_set = set(tuple(seq) for seq in target)
print("切分结果是否完全匹配目标：", result_set == target_set)
print("所有切分方式：")
for seq in result:
    print(seq)
