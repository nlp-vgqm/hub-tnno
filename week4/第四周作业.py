#week4
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
    def back_cut(str_back, path):
        if str_back == "":
            target.append(path[:])
            return
        # 尝试所有可能的切分
        for i in range(1, len(str_back) + 1):
            word = str_back[:i]
            # 如果当前词在词典中
            if word in Dict:
                path.append(word)
                back_cut(str_back[i:], path)
                path.pop()

    target = []
    back_cut(sentence,[])
    return target

#打印
for i, tar_list in enumerate(all_cut(sentence, Dict),1):
   print(i,tar_list)
