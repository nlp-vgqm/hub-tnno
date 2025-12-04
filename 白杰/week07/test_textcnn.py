import pandas as pd
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# 设置设备：优先用GPU，无则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# ===================== 1. 数据读取与预处理 =====================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^\u4e00-\u9fa5，。！？]', '', str(text)).strip()

# 读取数据集
df = pd.read_csv(
    '文本分类练习.csv',
    sep=',', quotechar='"', header=None, names=['label', 'text'],
    encoding='utf-8'
)

# 关键修复：将标签列强制转为数值类型（处理字符串/空值等异常）
# errors='coerce'：无法转换的数值转为NaN；fillna(0)：NaN填充为0
df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0)
# 确保标签是float类型（适配BCELoss）
df['label'] = df['label'].astype(float)

# 文本预处理
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'] != ""]  # 过滤空文本
print(f"数据集规模：{len(df)} 条")
print(f"标签分布：好评(1)={sum(df['label']==1)}, 差评(0)={sum(df['label']==0)}")

# ===================== 2. 构建词汇表（基于TF-IDF） =====================
tfidf = TfidfVectorizer(max_features=2000)
tfidf.fit(df['text'])
vocab = {word: idx + 1 for idx, word in enumerate(tfidf.vocabulary_)}  # 0留给padding
vocab_size = len(vocab) + 1
max_seq_len = max([len(text) for text in df['text']])  # 文本最大长度
print(f"词汇表大小：{vocab_size} | 文本最大长度：{max_seq_len}")

# ===================== 3. 划分训练/验证集 =====================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ===================== 4. 自定义数据集类 =====================
class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 文本转索引（字符级）
        text = self.texts.iloc[idx]
        tokens = [self.vocab.get(char, 0) for char in text]
        # 截断/补齐到max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [0] * (self.max_len - len(tokens))
        # 转tensor（先构造CPU tensor，再移到设备，避免多进程加载问题）
        tokens = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)
        return tokens.to(device), label.to(device)

# 构建DataLoader
batch_size = 16
train_dataset = TextCNNDataset(train_texts, train_labels, vocab, max_seq_len)
val_dataset = TextCNNDataset(val_texts, val_labels, vocab, max_seq_len)
# 补充：Windows系统下num_workers设为0，避免多进程数据加载报错
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ===================== 5. TextCNN模型定义 =====================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=256, filter_sizes=[2, 3, 4]):
        super(TextCNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 卷积层（多个卷积核尺寸）
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        # Dropout（防止过拟合）
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, max_len]
        embed_x = self.embedding(x)  # [batch_size, max_len, embed_dim]
        embed_x = embed_x.unsqueeze(1)  # [batch_size, 1, max_len, embed_dim]（添加通道维度）

        # 卷积+池化
        conv_outs = []
        for conv in self.convs:
            out = conv(embed_x)  # [batch_size, num_filters, max_len-fs+1, 1]
            out = nn.functional.relu(out.squeeze(3))  # [batch_size, num_filters, max_len-fs+1]
            out = nn.functional.max_pool1d(out, out.size(2)).squeeze(2)  # [batch_size, num_filters]
            conv_outs.append(out)

        # 拼接所有卷积核结果
        concat_out = torch.cat(conv_outs, dim=1)  # [batch_size, num_filters*3]
        concat_out = self.dropout(concat_out)
        # 分类输出
        out = self.fc(concat_out)  # [batch_size, 1]
        out = self.sigmoid(out)
        return out.squeeze(1)  # [batch_size]

# ===================== 6. 模型初始化 =====================
cnn_model = TextCNN(vocab_size=vocab_size).to(device)
# 优化器+损失函数
learning_rate = 1e-3  # 核心学习率
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()  # 二分类交叉熵

# ===================== 7. 模型训练 =====================
epochs = 10  # 训练轮数
best_acc = 0.0
print("\n开始训练TextCNN...")
for epoch in range(epochs):
    cnn_model.train()
    train_loss = 0.0
    for batch_idx, (texts, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # 清零梯度
        outputs = cnn_model(texts)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        train_loss += loss.item()

    # 每轮训练后验证
    cnn_model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for texts, labels in val_loader:
            outputs = cnn_model(texts)
            # 预测标签（>0.5为1，否则为0）
            preds = (outputs > 0.5).float().cpu().numpy()
            val_preds.extend(preds)
            val_trues.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_trues, val_preds)
    print(f"Epoch {epoch + 1}/{epochs} | 训练损失：{train_loss / len(train_loader):.4f} | 验证准确率：{val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc

# ===================== 8. 预测耗时测试 =====================
print("\n测试预测耗时...")
# 取100条验证集样本测试（不足100条则取全部）
test_sample_num = min(100, len(val_texts))
test_samples = val_texts[:test_sample_num].tolist()
test_labels = val_labels[:test_sample_num].tolist()
test_dataset = TextCNNDataset(
    pd.Series(test_samples), pd.Series(test_labels), vocab, max_seq_len
)
test_loader = DataLoader(test_dataset, batch_size=test_sample_num, shuffle=False, num_workers=0)

# 多次测试取平均
total_time = 0
test_times = 10
cnn_model.eval()
with torch.no_grad():
    # 先预加载一次，避免首次加载耗时干扰
    texts, _ = next(iter(test_loader))
    cnn_model(texts)
    # 正式测试
    for _ in range(test_times):
        start = time.time()
        cnn_model(texts)
        total_time += (time.time() - start)
avg_time = total_time / test_times

# ===================== 9. 输出核心参数 =====================
print("\n【TextCNN核心参数】")
print(f"学习率：{learning_rate}")
print(f"嵌入维度：128")
print(f"卷积核数量：256 | 卷积核尺寸：[2,3,4]")
print(f"最佳验证准确率：{best_acc:.4f}")
print(f"{test_sample_num}条样本平均预测耗时：{avg_time:.4f} 秒")
print(f"单条样本平均预测耗时：{avg_time / test_sample_num:.6f} 秒")