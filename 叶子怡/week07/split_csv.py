import pandas as pd

data = pd.read_csv('D:\\nlp2025\课件\week7 文本分类问题\pipline_homework\文本分类练习.csv')
# print(data.shape)  # (11987, 2)

# 打印前十行
# print(data.head(10))

# 查看标签类别及数目
# print(data.label.value_counts())
'''
0    7987
1    4000
'''

# 标签将总的dataframe分割为两份，一份为标签为1，一份为标签为0
groups = data.groupby(data.label)
data_true = groups.get_group(1)
data_false = groups.get_group(0)
# print(data_true.head(5))
# print(data_false.head(5))

# 打乱
data_true = data_true.sample(frac=1.0).reset_index(drop=True)
data_false = data_false.sample(frac=1.0).reset_index(drop=True)

# 测试集，按照大概8：1：1的比例取
test_true = data_true.iloc[:1600, :]
test_false = data_false.iloc[:800, :]
test_data = pd.concat([test_true, test_false], axis = 0, ignore_index=True).sample(frac=1.0)


# #验证集
# valid_true = data_true.iloc[800:1600, :]
# valid_false = data_false.iloc[400:800, :]
# valid_data = pd.concat([valid_true, valid_false], axis = 0, ignore_index=True).sample(frac=1)

#训练集
train_true = data_true.iloc[1600:, :]
train_false = data_false.iloc[800:, :]
train_data = pd.concat([train_true, train_false], axis = 0, ignore_index=True).sample(frac=1)

# 生成csv文件
# test_data.to_csv("test.csv", index=False, encoding='utf-8')
# valid_data.to_csv("val.csv", index=False, encoding='utf-8')
# train_data.to_csv("train.csv", index=False, encoding='utf-8')
