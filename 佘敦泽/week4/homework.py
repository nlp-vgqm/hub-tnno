#week3作业
# 实现句子全切分
from attr.validators import max_len

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

#TODO 实现全切分函数，输出根据字典能够切分出的所有的切分方式
# 使用递归、堆栈方式处理
def all_cut(sentence, Dict):
    # dict最大长度
    max_len = max([len(key) for key, value in Dict.items()])
    print(f'max_len: {max_len}')

    def recursion(start_index, current_result):
        if start_index >= len(sentence):
            target.append(current_result[:]) # 不能直接使用 current_result, 后面pop的时候会将这里的数据也清理掉(共用了底层内存)
            return

        for length in range(min(max_len, len(sentence) - start_index), 0, -1):
            sub_word = sentence[start_index: start_index + length]
            if sub_word in Dict:
                current_result.append(sub_word)
                recursion(start_index + length, current_result)
                current_result.pop() # 移除最后一个词

    target = []
    recursion(0, [])

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


if __name__ == '__main__':
    # all_cut(sentence, Dict)
    target = all_cut(sentence, Dict)
    print(target)