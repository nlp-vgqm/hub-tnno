import pandas as pd
import re
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置设备
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
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'] != ""]
print(f"数据集规模：{len(df)} 条")

# ===================== 2. 划分训练/验证集 =====================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ===================== 3. BERTTokenizer初始化 =====================
# 使用中文BERT预训练模型
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
max_seq_len = 128  # BERT最大长度512，这里用128足够（短文本）


# ===================== 4. 自定义数据集类 =====================
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        # 编码文本（BERT要求的格式）
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加[CLS]、[SEP]
            max_length=self.max_len,
            padding='max_length',  # 补齐到max_len
            truncation=True,  # 截断过长文本
            return_attention_mask=True,
            return_tensors='pt'  # 返回tensor
        )
        # 转device并展平（去掉batch维度）
        input_ids = encoding['input_ids'].flatten().to(device)
        attention_mask = encoding['attention_mask'].flatten().to(device)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long).to(device)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


# 构建DataLoader
batch_size = 8  # BERT显存占用大，batch_size调小
train_dataset = BERTDataset(train_texts, train_labels, tokenizer, max_seq_len)
val_dataset = BERTDataset(val_texts, val_labels, tokenizer, max_seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ===================== 5. BERT模型初始化 =====================
bert_model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 二分类：好评/差评
).to(device)

# 优化器（BERT推荐用AdamW）
learning_rate = 2e-5  # BERT核心学习率（不能大，否则冲掉预训练权重）
optimizer = AdamW(bert_model.parameters(), lr=learning_rate, eps=1e-8)

# ===================== 6. 模型训练 =====================
epochs = 3  # BERT训练轮数不用多，避免过拟合
best_acc = 0.0
print("\n开始训练BERT...")
for epoch in range(epochs):
    bert_model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # 前向传播
        outputs = bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['label']
        )
        loss = outputs.loss
        train_loss += loss.item()
        # 反向传播
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
        optimizer.step()

    # 每轮训练后验证
    bert_model.eval()
    val_preds = []
    val_trues = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = bert_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            # 取logits最大值作为预测标签
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_trues.extend(batch['label'].cpu().numpy())
    val_acc = accuracy_score(val_trues, val_preds)
    print(f"Epoch {epoch + 1}/{epochs} | 训练损失：{train_loss / len(train_loader):.4f} | 验证准确率：{val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc

# ===================== 7. 预测耗时测试 =====================
print("\n测试预测耗时...")
# 取100条验证集样本测试
test_samples = val_texts[:100].tolist()
test_labels = val_labels[:100].tolist()
test_dataset = BERTDataset(
    pd.Series(test_samples), pd.Series(test_labels), tokenizer, max_seq_len
)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 多次测试取平均
total_time = 0
test_times = 5  # BERT耗时久，少测几次
bert_model.eval()
with torch.no_grad():
    for _ in range(test_times):
        batch = next(iter(test_loader))
        start = time.time()
        bert_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        total_time += (time.time() - start)
avg_time = total_time / test_times

# ===================== 8. 输出核心参数 =====================
print("\n【BERT核心参数】")
print(f"预训练模型：{model_name}")
print(f"学习率：{learning_rate}")
print(f"最大序列长度：{max_seq_len}")
print(f"最佳验证准确率：{best_acc:.4f}")
print(f"100条样本平均预测耗时：{avg_time:.4f} 秒")
print(f"单条样本平均预测耗时：{avg_time / 100:.6f} 秒")