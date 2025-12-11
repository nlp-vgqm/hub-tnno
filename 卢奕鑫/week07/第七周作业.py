import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 读取数据
print("=" * 60)
print("加载数据集...")
df = pd.read_csv('文本分类练习.csv')
print(f"数据集形状: {df.shape}")

# ==================== 数据分析 ====================
print("\n" + "=" * 60)
print("数据分析")
print("=" * 60)

# 1. 正负样本数
positive_count = df['label'].sum()
negative_count = len(df) - positive_count
print(f"正样本数(好评): {positive_count}")
print(f"负样本数(差评): {negative_count}")
print(f"正负样本比例: {positive_count / negative_count:.2f}:1")

# 2. 文本长度分析
df['text_length'] = df['review'].apply(len)
print(f"\n文本平均长度: {df['text_length'].mean():.2f} 字符")
print(f"文本长度分布: min={df['text_length'].min()}, "
      f"max={df['text_length'].max()}, median={df['text_length'].median()}")

# ==================== 数据预处理 ====================
print("\n" + "=" * 60)
print("数据预处理")
print("=" * 60)

# 划分训练集和验证集
X = df['review'].values
y = df['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")

# 方法1：使用词袋模型（适合MLP）
print("\n方法1: 使用TF-IDF特征（适合MLP模型）...")
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_tfidf = TfidfVectorizer(max_features=3000, max_df=0.95, min_df=2)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer_tfidf.transform(X_val).toarray()
print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")

# 方法2：使用序列表示（适合CNN/LSTM/Transformer）
print("\n方法2: 使用序列表示（适合CNN/LSTM/Transformer模型）...")

# 创建词汇表
vocab = {}
word_to_idx = {}
idx_to_word = {}
vocab_size = 5000  # 限制词汇表大小

# 构建词汇表
print("构建词汇表...")
from collections import Counter

all_words = []
for text in X_train:
    all_words.extend(str(text).split())

word_counts = Counter(all_words)
common_words = word_counts.most_common(vocab_size - 2)  # 保留位置给PAD和UNK

# 建立词汇映射
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1
idx_to_word[0] = '<PAD>'
idx_to_word[1] = '<UNK>'

for idx, (word, _) in enumerate(common_words, start=2):
    word_to_idx[word] = idx
    idx_to_word[idx] = word


# 文本转序列
def text_to_sequence(text, max_len=100):
    words = str(text).split()
    sequence = []
    for word in words[:max_len]:
        sequence.append(word_to_idx.get(word, word_to_idx['<UNK>']))
    # 填充或截断
    if len(sequence) < max_len:
        sequence += [word_to_idx['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence


# 转换所有文本
max_sequence_len = 100
X_train_seq = np.array([text_to_sequence(text, max_sequence_len) for text in X_train])
X_val_seq = np.array([text_to_sequence(text, max_sequence_len) for text in X_val])

print(f"序列长度: {max_sequence_len}")
print(f"词汇表大小: {len(word_to_idx)}")
print(f"训练集序列形状: {X_train_seq.shape}")
print(f"验证集序列形状: {X_val_seq.shape}")

# ==================== 深度学习模型 ====================
print("\n" + "=" * 60)
print("构建深度学习模型")
print("=" * 60)


# 1. MLP模型（使用TF-IDF特征）
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=2):
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 2. CNN模型（使用序列特征）
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super(CNNModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 卷积层 - 使用不同尺寸的卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 128, kernel_size=(k, embed_dim))
            for k in [3, 4, 5]
        ])

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc = nn.Linear(128 * 3, num_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]

        # 不同尺寸的卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(embedded)).squeeze(3)  # [batch_size, 128, seq_len-k+1]
            pool_out = torch.max(conv_out, dim=2)[0]  # [batch_size, 128]
            conv_outputs.append(pool_out)

        # 拼接特征
        cat = torch.cat(conv_outputs, dim=1)  # [batch_size, 128*3]

        # Dropout和全连接
        cat = self.dropout(cat)
        output = self.fc(cat)

        return output


# 3. LSTM模型（使用序列特征）
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(LSTMModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM层
        self.lstm = nn.LSTM(embed_dim, hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2,
                            dropout=0.3)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]

        # 全连接
        output = self.fc(last_output)

        return output


# 4. 简单的Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_classes=2):
        super(TransformerModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, max_sequence_len, embed_dim) * 0.01)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.3,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 嵌入和位置编码
        embedded = self.embedding(x) + self.pos_encoder

        # 创建注意力掩码（忽略PAD标记）
        mask = (x == 0)

        # Transformer编码
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)

        # 取第一个位置的输出（类似BERT的[CLS]）
        first_token = encoded[:, 0, :]

        # 分类
        output = self.fc(first_token)

        return output


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, model_name='Model'):
    print(f"\n训练{model_name}...")
    model = model.to(device)

    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch + 1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    training_time = time.time() - start_time

    return model, history, training_time


# 评估函数 - 为不同模型类型分别处理
def evaluate_model(model, X_val, y_val, device, model_type='mlp'):
    model.eval()

    if model_type == 'mlp':
        # MLP使用FloatTensor
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    else:
        # 其他模型使用LongTensor（序列数据）
        val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predictions = outputs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return accuracy, f1, np.array(all_predictions)


# 模型预测速度测试
def test_prediction_speed(model, test_data, device, model_name, model_type='mlp', num_tests=50):
    print(f"测试{model_name}预测速度...")
    model.eval()

    # 准备测试数据
    if model_type == 'mlp':
        test_tensor = torch.FloatTensor(test_data).to(device)
    else:
        test_tensor = torch.LongTensor(test_data).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_tensor[:10])

    # 实际测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_tests):
            outputs = model(test_tensor)

    total_time = time.time() - start_time
    avg_time_per_sample = (total_time / num_tests) / len(test_tensor) * 1000  # 毫秒

    print(f"平均预测时间: {avg_time_per_sample:.2f} 毫秒/样本")
    return avg_time_per_sample


# ==================== 主训练流程 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

results = []

# 1. 训练MLP模型（使用TF-IDF特征）
print("\n" + "=" * 60)
print("1. 训练MLP模型")
print("=" * 60)

# 准备数据
X_train_mlp = torch.FloatTensor(X_train_tfidf)
X_val_mlp = torch.FloatTensor(X_val_tfidf)
y_train_tensor = torch.LongTensor(y_train)
y_val_tensor = torch.LongTensor(y_val)

train_dataset_mlp = TensorDataset(X_train_mlp, y_train_tensor)
val_dataset_mlp = TensorDataset(X_val_mlp, y_val_tensor)
train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=32, shuffle=True)
val_loader_mlp = DataLoader(val_dataset_mlp, batch_size=32, shuffle=False)

# 创建和训练MLP模型
mlp_model = MLPModel(input_dim=X_train_tfidf.shape[1])
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trained_mlp, mlp_history, mlp_train_time = train_model(
    mlp_model, train_loader_mlp, val_loader_mlp, optimizer, criterion,
    device, epochs=10, model_name='MLP'
)

# 评估MLP
mlp_accuracy, mlp_f1, mlp_predictions = evaluate_model(
    trained_mlp, X_val_tfidf, y_val, device, model_type='mlp'
)
mlp_speed = test_prediction_speed(trained_mlp, X_val_tfidf, device, 'MLP', model_type='mlp')

# 保存MLP结果
results.append({
    '模型': 'MLP',
    '训练时间(秒)': round(mlp_train_time, 2),
    '验证准确率(%)': round(mlp_accuracy * 100, 2),
    'F1分数': round(mlp_f1, 4),
    '预测速度(ms/样本)': round(mlp_speed, 2),
    '参数量(M)': round(sum(p.numel() for p in mlp_model.parameters()) / 1e6, 3)
})

# 2. 训练CNN模型（使用序列特征）
print("\n" + "=" * 60)
print("2. 训练CNN模型")
print("=" * 60)

# 准备数据
train_dataset_cnn = TensorDataset(torch.LongTensor(X_train_seq), torch.LongTensor(y_train))
val_dataset_cnn = TensorDataset(torch.LongTensor(X_val_seq), torch.LongTensor(y_val))
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=32, shuffle=True)
val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=32, shuffle=False)

# 创建和训练CNN模型
cnn_model = CNNModel(vocab_size=len(word_to_idx))
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trained_cnn, cnn_history, cnn_train_time = train_model(
    cnn_model, train_loader_cnn, val_loader_cnn, optimizer, criterion,
    device, epochs=10, model_name='CNN'
)

# 评估CNN
cnn_accuracy, cnn_f1, cnn_predictions = evaluate_model(
    trained_cnn, X_val_seq, y_val, device, model_type='seq'
)
cnn_speed = test_prediction_speed(trained_cnn, X_val_seq, device, 'CNN', model_type='seq')

# 保存CNN结果
results.append({
    '模型': 'CNN',
    '训练时间(秒)': round(cnn_train_time, 2),
    '验证准确率(%)': round(cnn_accuracy * 100, 2),
    'F1分数': round(cnn_f1, 4),
    '预测速度(ms/样本)': round(cnn_speed, 2),
    '参数量(M)': round(sum(p.numel() for p in cnn_model.parameters()) / 1e6, 3)
})

# 3. 训练LSTM模型（使用序列特征）
print("\n" + "=" * 60)
print("3. 训练LSTM模型")
print("=" * 60)

# 创建和训练LSTM模型
lstm_model = LSTMModel(vocab_size=len(word_to_idx))
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trained_lstm, lstm_history, lstm_train_time = train_model(
    lstm_model, train_loader_cnn, val_loader_cnn, optimizer, criterion,
    device, epochs=10, model_name='LSTM'
)

# 评估LSTM
lstm_accuracy, lstm_f1, lstm_predictions = evaluate_model(
    trained_lstm, X_val_seq, y_val, device, model_type='seq'
)
lstm_speed = test_prediction_speed(trained_lstm, X_val_seq, device, 'LSTM', model_type='seq')

# 保存LSTM结果
results.append({
    '模型': 'LSTM',
    '训练时间(秒)': round(lstm_train_time, 2),
    '验证准确率(%)': round(lstm_accuracy * 100, 2),
    'F1分数': round(lstm_f1, 4),
    '预测速度(ms/样本)': round(lstm_speed, 2),
    '参数量(M)': round(sum(p.numel() for p in lstm_model.parameters()) / 1e6, 3)
})

# 4. 训练Transformer模型
print("\n" + "=" * 60)
print("4. 训练Transformer模型")
print("=" * 60)

# 创建和训练Transformer模型
transformer_model = TransformerModel(vocab_size=len(word_to_idx))
optimizer = optim.Adam(transformer_model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

trained_transformer, transformer_history, transformer_train_time = train_model(
    transformer_model, train_loader_cnn, val_loader_cnn, optimizer, criterion,
    device, epochs=10, model_name='Transformer'
)

# 评估Transformer
transformer_accuracy, transformer_f1, transformer_predictions = evaluate_model(
    trained_transformer, X_val_seq, y_val, device, model_type='seq'
)
transformer_speed = test_prediction_speed(trained_transformer, X_val_seq, device, 'Transformer', model_type='seq')

# 保存Transformer结果
results.append({
    '模型': 'Transformer',
    '训练时间(秒)': round(transformer_train_time, 2),
    '验证准确率(%)': round(transformer_accuracy * 100, 2),
    'F1分数': round(transformer_f1, 4),
    '预测速度(ms/样本)': round(transformer_speed, 2),
    '参数量(M)': round(sum(p.numel() for p in transformer_model.parameters()) / 1e6, 3)
})

# ==================== 结果展示 ====================
print("\n" + "=" * 60)
print("模型性能对比总结")
print("=" * 60)

# 创建结果DataFrame
results_df = pd.DataFrame(results)

# 格式化输出
print("\n" + "=" * 80)
print("模型性能对比表")
print("=" * 80)
print(f"{'模型':<15} {'准确率(%)':<12} {'F1分数':<12} {'训练时间(s)':<12} {'预测速度(ms)':<12} {'参数量(M)':<12}")
print("-" * 80)

for _, row in results_df.iterrows():
    print(f"{row['模型']:<15} {row['验证准确率(%)']:<12.2f} {row['F1分数']:<12.4f} "
          f"{row['训练时间(秒)']:<12.2f} {row['预测速度(ms/样本)']:<12.2f} {row['参数量(M)']:<12.3f}")

print("=" * 80)

# 找到最佳模型
best_acc_idx = results_df['验证准确率(%)'].idxmax()
best_f1_idx = results_df['F1分数'].idxmax()
fastest_idx = results_df['预测速度(ms/样本)'].idxmin()

best_acc_model = results_df.loc[best_acc_idx]
best_f1_model = results_df.loc[best_f1_idx]
fastest_model = results_df.loc[fastest_idx]

print("\n" + "=" * 60)
print("最佳模型分析")
print("=" * 60)
print(f"🏆 准确率最高: {best_acc_model['模型']} - {best_acc_model['验证准确率(%)']}%")
print(f"🎯 F1分数最高: {best_f1_model['模型']} - {best_f1_model['F1分数']}")
print(f"⚡ 预测速度最快: {fastest_model['模型']} - {fastest_model['预测速度(ms/样本)']}ms/样本")

# 打印分类报告（选择最佳模型）
print(f"\n📊 {best_acc_model['模型']} 详细分类报告:")
if best_acc_model['模型'] == 'MLP':
    predictions = mlp_predictions
elif best_acc_model['模型'] == 'CNN':
    predictions = cnn_predictions
elif best_acc_model['模型'] == 'LSTM':
    predictions = lstm_predictions
else:
    predictions = transformer_predictions

print(classification_report(y_val, predictions, target_names=['差评', '好评']))

# 打印混淆矩阵
print("混淆矩阵:")
cm = confusion_matrix(y_val, predictions)
print(f"TP(真正例): {cm[1, 1]}  FP(假正例): {cm[0, 1]}")
print(f"FN(假负例): {cm[1, 0]}  TN(真负例): {cm[0, 0]}")

# ==================== 总结分析 ====================
print("\n" + "=" * 60)
print("模型特点总结与建议")
print("=" * 60)
print("""
📈 模型特点分析：

1. MLP（多层感知机）:
   - ✅ 优点: 训练和预测速度最快，实现简单
   - ❌ 缺点: 无法捕捉序列信息和上下文关系
   - 📊 适用: 对速度要求高，文本特征明确的任务

2. CNN（卷积神经网络）:
   - ✅ 优点: 能有效捕捉局部特征和短语模式
   - ❌ 缺点: 对长距离依赖处理有限
   - 📊 适用: 短文本情感分析，关键词识别

3. LSTM（长短期记忆网络）:
   - ✅ 优点: 擅长处理序列依赖，记忆长期信息
   - ❌ 缺点: 训练较慢，参数量大
   - 📊 适用: 长文本分析，需要考虑上下文的场景

4. Transformer:
   - ✅ 优点: 并行计算，注意力机制强大
   - ❌ 缺点: 需要大量数据，训练时间长
   - 📊 适用: 复杂语义理解，大规模文本分类

💡 推荐建议:
- 如果追求速度: 选择 MLP
- 如果追求平衡: 选择 CNN 或 LSTM
- 如果数据量大: 选择 Transformer
- 电商评论分类: 推荐 CNN 或 LSTM
""")

# 保存结果
print("\n" + "=" * 60)
print("保存实验结果")
print("=" * 60)

# 保存结果到CSV
results_df.to_csv('电商评论分类_模型对比结果.csv', index=False, encoding='utf-8-sig')
print("✅ 结果已保存到 '电商评论分类_模型对比结果.csv'")

# 保存详细报告
with open('电商评论分类_实验报告.txt', 'w', encoding='utf-8') as f:
    f.write("电商评论分类实验报告\n")
    f.write("=" * 50 + "\n\n")

    f.write("一、实验基本信息\n")
    f.write(f"- 数据集: {len(df)} 条评论\n")
    f.write(f"- 训练集: {len(X_train)} 条\n")
    f.write(f"- 验证集: {len(X_val)} 条\n")
    f.write(f"- 正样本: {positive_count} 条\n")
    f.write(f"- 负样本: {negative_count} 条\n")
    f.write(f"- 词汇表大小: {len(word_to_idx)}\n")
    f.write(f"- 序列长度: {max_sequence_len}\n")
    f.write(f"- TF-IDF特征维度: {X_train_tfidf.shape[1]}\n\n")

    f.write("二、模型性能对比\n")
    f.write(results_df.to_string(index=False) + "\n\n")

    f.write("三、最佳模型\n")
    f.write(f"1. 准确率最高: {best_acc_model['模型']} ({best_acc_model['验证准确率(%)']}%)\n")
    f.write(f"2. F1分数最高: {best_f1_model['模型']} ({best_f1_model['F1分数']})\n")
    f.write(f"3. 预测最快: {fastest_model['模型']} ({fastest_model['预测速度(ms/样本)']}ms/样本)\n\n")

    f.write("四、结论与建议\n")
    f.write("1. 对于电商评论分类任务，CNN和LSTM模型表现较好\n")
    f.write("2. MLP模型速度最快，适合实时应用\n")
    f.write("3. Transformer模型在足够数据下潜力最大\n")
    f.write("4. 推荐实际应用中使用CNN模型，平衡性能与速度\n")

print("✅ 详细报告已保存到 '电商评论分类_实验报告.txt'")

# 最终总结表格
print("\n" + "=" * 60)
print("最终对比结果（简化版）")
print("=" * 60)

print("\n┌─────────────┬────────────┬──────────┬────────────┬────────────┐")
print("│    模型     │  准确率    │  F1分数  │  预测速度  │  参数量    │")
print("├─────────────┼────────────┼──────────┼────────────┼────────────┤")

for _, row in results_df.iterrows():
    # 标记最佳值
    acc_str = f"{row['验证准确率(%)']:.1f}%"
    f1_str = f"{row['F1分数']:.3f}"
    speed_str = f"{row['预测速度(ms/样本)']:.2f}ms"
    param_str = f"{row['参数量(M)']:.2f}M"

    if row['验证准确率(%)'] == results_df['验证准确率(%)'].max():
        acc_str = "★" + acc_str
    if row['F1分数'] == results_df['F1分数'].max():
        f1_str = "★" + f1_str
    if row['预测速度(ms/样本)'] == results_df['预测速度(ms/样本)'].min():
        speed_str = "⚡" + speed_str

    print(f"│ {row['模型']:^11} │ {acc_str:^10} │ {f1_str:^8} │ {speed_str:^10} │ {param_str:^10} │")

print("└─────────────┴────────────┴──────────┴────────────┴────────────┘")
print("\n注: ★表示该项指标最佳，⚡表示预测速度最快")
print("\n🎉 实验完成！所有结果已保存到文件。")

# 输出一些示例预测
print("\n" + "=" * 60)
print("示例预测")
print("=" * 60)

# 选择几个示例文本
sample_texts = X_val[:5]
sample_labels = y_val[:5]

print("前5个验证集样本的预测结果:")
print("-" * 60)

for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
    # 使用最佳模型进行预测
    if best_acc_model['模型'] == 'MLP':
        # 转换为TF-IDF特征
        text_tfidf = vectorizer_tfidf.transform([text]).toarray()
        input_tensor = torch.FloatTensor(text_tfidf).to(device)
        with torch.no_grad():
            output = trained_mlp(input_tensor)
            predicted = output.argmax().item()
    else:
        # 转换为序列
        text_seq = text_to_sequence(text, max_sequence_len)
        input_tensor = torch.LongTensor([text_seq]).to(device)

        if best_acc_model['模型'] == 'CNN':
            with torch.no_grad():
                output = trained_cnn(input_tensor)
                predicted = output.argmax().item()
        elif best_acc_model['模型'] == 'LSTM':
            with torch.no_grad():
                output = trained_lstm(input_tensor)
                predicted = output.argmax().item()
        else:
            with torch.no_grad():
                output = trained_transformer(input_tensor)
                predicted = output.argmax().item()

    true_label_str = "好评" if true_label == 1 else "差评"
    predicted_str = "好评" if predicted == 1 else "差评"
    correct = "✓" if predicted == true_label else "✗"

    print(f"样本 {i + 1}:")
    print(f"  文本: {text[:50]}...")
    print(f"  真实标签: {true_label_str}")
    print(f"  预测标签: {predicted_str} {correct}")
    print("-" * 40)
