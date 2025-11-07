#week3作业

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
    index_str_dict = {}
    for i in range(0, len(sentence)):
        index_str_dict[i] = find_word_in_dict(sentence[i:i + 1], Dict)
    for i in range(0, len(sentence) - 1):
        # 两两合并，0下标的分词依次和1下标合并，再和2下标合并，依次合并到最后一个下标的分词
        # 最后剩下的0下标的分词则为所有切分好的结果的总集合
        merge_two_list(index_str_dict, i + 1)
    return index_str_dict.get(0)

# 将词典里以word开头的词都统计好，并将下标和词组合起来放到一个新字典里
# 形成如下数据结构：
# {0: [['经常'], ['经']], 1: [['常']], 2: [['有'], ['有意见']], 3: [['意见'], ['意']], 4: [['见'], ['见分歧']], 5: [['分歧'], ['分']], 6: [['歧']]}
def find_word_in_dict(word, Dict):
    index_list = []
    for str in Dict.keys():
        if str.startswith(word):
            list = []
            list.append(str)
            index_list.append(list)
    return index_list

# 合并字典里下标为0和1的分词集合，依次进行两两合并
def merge_two_list(index_str_dict, increa):
    i = 0
    list_i = []
    j = i + increa
    list1 = index_str_dict.get(i)
    list2 = index_str_dict.get(j)
    for l_i in list1:
        if i + list_word_len(l_i) > j:
            list_i.append(l_i)
            continue
        for l_j in list2:
            list_i.append(list_union(l_i, l_j))
    index_str_dict[i] = list_i
    del index_str_dict[j]

# 将两个集合里的元素合并到一个集合里
def list_union(list1, list2):
    list = []
    for s in list1:
        list.append(s)
    for s in list2:
        list.append(s)
    return list

# 计算集合里所有元素长度的总和
def list_word_len(list):
    word_len = 0
    for word in list:
        word_len += len(word)
    return word_len

if __name__ == "__main__":
    target = all_cut(sentence, Dict)
    print('全分词后结果总数：', len(target), "  具体分词如下：")
    for list in target:
        print(list)
