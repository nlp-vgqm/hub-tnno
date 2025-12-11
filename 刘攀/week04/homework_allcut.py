# week3作业

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
    word_set = set(Dict.keys())  # 字典词转为集合，O(1)查询效率
    max_word_len = max(len(word) for word in word_set) if word_set else 0  # 最长词长度（优化遍历范围）
    result = []  # 存储所有切分结果

    # 递归回溯函数：start=当前处理起始索引，path=当前切分路径
    def backtrack(start, path):
        # 终止条件：起始索引到达句尾，说明找到一种完整切分
        if start == len(sentence):
            result.append(path.copy())  # 复制路径，避免回溯修改
            return

        for end in range(start + 1, len(sentence) + 1):
            substr = sentence[start:end]  # 提取当前子串
            if substr in word_set:  # 若子串是字典中的有效词
                path.append(substr)  # 加入当前路径
                backtrack(end, path)  # 递归处理剩余子串（起始索引更新为end）
                path.pop()  # 回溯：撤销当前选择，尝试下一种切分

    # 从句子开头（索引0）启动递归
    backtrack(0, path=[])
    return result

# 目标输出;顺序不重要
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

# 执行流程
if __name__ == "__main__":
    # 1. 执行全切分
    result = all_cut(sentence, Dict)


    # 2. 验证结果（排序后对比，消除顺序影响）
    def sort_cuts(cut_list):
        return sorted([tuple(cut) for cut in cut_list])


    target_sorted = sort_cuts(target)
    result_sorted = sort_cuts(result)

    # 断言验证
    assert result_sorted == target_sorted, "切分结果与目标不一致！"
    print("全切分成功！共得到 {} 种切分方式：\n".format(len(result)))

    # 输出所有切分结果
    for i, cut in enumerate(result, 1):
        print(f"{i:2d}. {cut}")