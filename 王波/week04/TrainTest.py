# 作业
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
        def backtrack(start, path, result):
                # 如果已经处理到句子末尾，将当前路径添加到结果中
                if start == len(sentence):
                        result.append(path[:])
                        return  # 返回到调用它的地方（上一个递归层）

                # 尝试所有可能的切分
                for end in range(start + 1, len(sentence) + 1):
                        word = sentence[start:end]
                        # 如果当前词在字典中，继续递归
                        if word in Dict:
                                path.append(word)
                                backtrack(end, path, result)
                                # 会有两种return，显式return：触发终止条件；隐式return：for循环结束
                                # 返回顺序：递归返回的顺序与调用顺序相反（后进先出）
                                path.pop()  # 回溯:pop(),列表的一个内置方法，用于移除并返回列表中的最后一个元素。

        result = []
        backtrack(0, [], result)
        return result

# 测试函数
target = all_cut(sentence, Dict)

# 输出结果验证
print("找到的切分方式数量:", len(target))
for i, segmentation in enumerate(target, 1):
    print(f"{i:2d}: {segmentation}")

# 与目标输出对比
w_target = [
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
    ['经', '常', '有', '意', '见', '分', '歧']]
expected_count = len(w_target)



print(f"\n预期切分方式数量: {expected_count}")
print(f"实际找到数量: {len(target)}")
print(f"是否匹配: {len(target) == expected_count}")

