import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ===================== 1. 数据读取与预处理 =====================
def preprocess_text(text):
    """文本预处理：仅保留中文和核心标点，去除无关字符"""
    if pd.isna(text):
        return ""
    # 保留中文、，。！？，去除其他字符
    return re.sub(r'[^\u4e00-\u9fa5，。！？]', '', str(text)).strip()

# 读取数据集（适配你的CSV格式：无表头，第一列label，第二列text）
df = pd.read_csv(
    '文本分类练习.csv',
    sep=',',
    quotechar='"',  # 处理带引号的文本
    header=None,
    names=['label', 'text'],
    encoding='utf-8'  # 确保中文编码正确
)

# 关键修复：将标签列强制转为数值类型（处理字符串/空值等异常）
# errors='coerce'：无法转换的数值转为NaN；fillna(0)：NaN填充为0
df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0)
# 确保标签是整数类型（好评1/差评0）
df['label'] = df['label'].astype(int)

# 预处理文本
df['text'] = df['text'].apply(preprocess_text)
# 过滤空文本
df = df[df['text'] != ""]

# 数据校验：打印标签唯一值，排查异常
print(f"标签列唯一值：{df['label'].unique()}")
print(f"数据集规模：{len(df)} 条")
print(f"好评数（label=1）：{(df['label']==1).sum()} | 差评数（label=0）：{(df['label']==0).sum()}")

# ===================== 2. 划分训练/验证集 =====================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'],
    test_size=0.2,  # 20%验证集
    random_state=42,  # 固定随机种子，结果可复现
    stratify=df['label']  # 保证训练/验证集标签分布一致
)

# ===================== 3. TF-IDF特征编码 =====================
tfidf = TfidfVectorizer(
    max_features=2000,  # 保留Top2000高频词
    ngram_range=(1,2)   # 单字+双字特征，提升效果
)
# 仅在训练集拟合（避免数据泄露）
train_features = tfidf.fit_transform(train_texts)
val_features = tfidf.transform(val_texts)

# ===================== 4. 模型训练 =====================
nb_model = MultinomialNB(alpha=1.0)  # alpha是平滑参数，默认1.0
nb_model.fit(train_features, train_labels)

# ===================== 5. 模型评估 =====================
# 验证集准确率
val_pred = nb_model.predict(val_features)
val_acc = accuracy_score(val_labels, val_pred)
print(f"\n朴素贝叶斯验证集准确率：{val_acc:.4f}")

# 额外：打印混淆矩阵，更直观查看分类效果
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(val_labels, val_pred)
print(f"\n混淆矩阵：")
print(f"真实差评(0) - 预测差评(0)：{cm[0][0]} | 真实差评(0) - 预测好评(1)：{cm[0][1]}")
print(f"真实好评(1) - 预测差评(0)：{cm[1][0]} | 真实好评(1) - 预测好评(1)：{cm[1][1]}")

# ===================== 6. 预测耗时测试 =====================
# 测试100条样本的预测耗时（模拟批量预测）
test_sample_num = min(100, len(val_texts))  # 避免验证集不足100条的情况
test_samples = val_texts[:test_sample_num].tolist()
test_features = tfidf.transform(test_samples)

# 多次测试取平均（避免偶然误差）
total_time = 0
test_times = 10
for _ in range(test_times):
    start = time.time()
    nb_model.predict(test_features)
    total_time += (time.time() - start)
avg_time = total_time / test_times

print(f"\n{test_sample_num}条样本平均预测耗时：{avg_time:.4f} 秒")
print(f"单条样本平均预测耗时：{avg_time/test_sample_num:.6f} 秒")

# ===================== 7. 输出核心参数 =====================
print("\n【朴素贝叶斯核心参数】")
print(f"学习率：无（朴素贝叶斯无梯度下降，无学习率）")
print(f"特征维度：{train_features.shape[1]}")
print(f"平滑参数alpha：{nb_model.alpha}")