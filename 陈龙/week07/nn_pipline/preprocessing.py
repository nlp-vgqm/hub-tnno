import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 读取数据
data = pd.read_csv(r"../文本分类练习.csv")
data = data.dropna()

print("原始数据信息:")
print(f"数据形状: {data.shape}")
print("标签分布:")
print(data['label'].value_counts())

# 创建保存目录（如果不存在）
save_dir = r"../data2"
os.makedirs(save_dir, exist_ok=True)
print(f"确保目录存在: {save_dir}")

# 将数据集分为训练集和测试集，使用stratify保证01样本分布均匀
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

print("\n分割后数据信息:")
print(f"训练集形状: {train_data.shape}")
print(f"测试集形状: {test_data.shape}")

print("\n训练集标签分布:")
print(train_data['label'].value_counts())

print("\n测试集标签分布:")
print(test_data['label'].value_counts())

# 保存数据集
train_path = os.path.join(save_dir, "train.csv")
test_path = os.path.join(save_dir, "test.csv")

train_data.to_csv(train_path, index=False, encoding='utf-8')
test_data.to_csv(test_path, index=False, encoding='utf-8')

print(f"\n数据集已成功保存!")
print(f"训练集: {train_path}")
print(f"测试集: {test_path}")