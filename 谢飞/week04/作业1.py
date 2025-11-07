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

def all_cut(sentence, Dict):
    """
    中文分词全切分递归实现
    """
    # 基准情况：当句子为空时，返回包含一个空列表的列表
    if len(sentence) == 0:
        return [[]]

    target = []

    # 遍历词典中的每个词
    for word in Dict:
        # 如果句子以当前词开头
        if sentence.startswith(word):
            # 获取剩余部分
            remaining = sentence[len(word):]
            # 递归处理剩余部分
            remaining_cuts = all_cut(remaining, Dict)
            # 将当前词与所有剩余部分的分词结果组合
            for cut in remaining_cuts:
                target.append([word] + cut)

    return target

# 测试
target = all_cut(sentence, Dict)
print(f"总共有 {len(target)} 种切分方式:")
for i, seg in enumerate(target, 1):
    print(f"{i}: {seg}")

print("\n与目标输出对比:")
expected = [
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

print(f"预期结果数量: {len(expected)}")
print(f"实际结果数量: {len(target)}")

# 检查是否包含所有预期结果
for exp in expected:
    if exp not in target:
        print(f"缺失: {exp}")