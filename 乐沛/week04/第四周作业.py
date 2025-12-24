def all_cut(sentence, Dict):
    sent_list = []  # 存储所有切分结果

    def frist_cut(sentence, current_cut):
        # 将字典只保留键
        keys = list(Dict.keys())

        # 如果句子为空，说明已经完成一次切分
        if sentence == '':
            sent_list.append(current_cut.copy())  # 保存当前切分结果
            return

        # 从长到短尝试所有可能的切分
        for i in range(len(sentence), 0, -1):  # 从最长开始尝试
            word = sentence[:i]
            if word in keys:
                # 添加当前词到切分路径
                current_cut.append(word)
                # 递归处理剩余部分
                frist_cut(sentence[i:], current_cut)
                # 回溯：移除当前词，尝试其他可能性
                current_cut.pop()


    frist_cut(sentence, [])
    return sent_list


if __name__ == '__main__':
    sentence = "经常有意见分歧"
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

    results = all_cut(sentence, Dict)

    print(results)
