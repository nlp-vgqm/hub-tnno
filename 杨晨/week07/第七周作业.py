import json
import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# 1. 数据加载与预处理
class TextDataset(Dataset):
    def __init__(self, file_path, max_length=100):
        self.data = []
        self.labels = []
        self.label2id = {}
        self.id2label = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = ' '.join(jieba.cut(item['text']))
                label = item['tag']
                
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                    self.id2label[len(self.id2label)] = label
                
                self.data.append(text)
                self.labels.append(self.label2id[label])
        
        # 数据分析（参考PPT第52页）
        print(f"数据集: {file_path}")
        print(f"样本数量: {len(self.data)}")
        print(f"类别分布: {Counter(self.labels)}")
        
        # TF-IDF特征提取（参考PPT第35页）
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.features = self.vectorizer.fit_transform(self.data).toarray()
        
        # 文本长度分析（参考PPT第52页）
        lengths = [len(text.split()) for text in self.data]
        print(f"平均文本长度: {np.mean(lengths):.2f}")
        print(f"最大文本长度: {max(lengths)}")
        print(f"最小文本长度: {min(lengths)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text': self.data[idx],
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 2. 模型实现（基于PPT中的模型架构）
class TextCNN(nn.Module):
    """TextCNN模型实现（参考PPT第39页）"""
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        # 卷积层（参考PPT第26页）
        self.conv1 = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, 100, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(1, 100, kernel_size=5, padding=1)
        
        # 池化层（参考PPT第26页）
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 分类层
        self.fc = nn.Linear(300, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 添加通道维度 [batch, 1, features]
        x = x.unsqueeze(1)
        
        # 多尺度卷积（参考PPT第39页）
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(x))
        conv3 = self.relu(self.conv3(x))
        
        # 池化
        pool1 = self.pool(conv1).squeeze(2)
        pool2 = self.pool(conv2).squeeze(2)
        pool3 = self.pool(conv3).squeeze(2)
        
        # 特征拼接
        concat = torch.cat((pool1, pool2, pool3), dim=1)
        concat = self.dropout(concat)
        
        # 分类
        logits = self.fc(concat)
        return logits

class TextRNN(nn.Module):
    """TextRNN模型实现（参考PPT第36页）"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super(TextRNN, self).__init__()
        # RNN层（参考PPT第25页）
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 分类层
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 添加序列维度 [batch, seq=1, features]
        x = x.unsqueeze(1)
        
        # RNN处理（参考PPT第36页）
        output, _ = self.rnn(x)
        
        # 取最后一个时间步的输出
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 分类
        logits = self.fc(last_output)
        return logits

class NaiveBayes:
    """朴素贝叶斯分类器（参考PPT第15-17页）"""
    def __init__(self):
        self.class_probs = None
        self.feature_probs = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 计算类先验概率（参考PPT第17页）
        self.class_probs = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_probs[i] = np.sum(y == c) / n_samples
        
        # 计算特征条件概率（参考PPT第17页）
        self.feature_probs = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            class_mask = (y == c)
            class_features = X[class_mask]
            
            # 使用拉普拉斯平滑
            self.feature_probs[i] = (class_features.sum(axis=0) + 1) / (class_features.sum() + n_features)
    
    def predict(self, X):
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        
        for i in range(len(self.classes)):
            # 对数似然计算（参考PPT第17页）
            class_log_prob = np.log(self.class_probs[i])
            feature_log_prob = np.sum(np.log(self.feature_probs[i]) * X, axis=1)
            log_probs[:, i] = class_log_prob + feature_log_prob
        
        return np.argmax(log_probs, axis=1)

# 3. 实验主函数
def main():
    # 加载数据（参考PPT第52页）
    train_dataset = TextDataset('train_tag_news.json')
    valid_dataset = TextDataset('valid_tag_news.json')
    
    # 创建数据加载器（参考PPT第34页）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    # 获取特征维度
    input_dim = train_dataset.features.shape[1]
    num_classes = len(train_dataset.label2id)
    
    # 模型初始化（实现三种模型）
    models = {
        "TextCNN": TextCNN(input_dim, num_classes),
        "TextRNN": TextRNN(input_dim, 128, num_classes),
        "NaiveBayes": NaiveBayes()
    }
    
    # 训练朴素贝叶斯（参考PPT第17页）
    train_features = np.array([item['features'].numpy() for item in train_dataset])
    train_labels = np.array([item['label'].item() for item in train_dataset])
    models['NaiveBayes'].fit(train_features, train_labels)
    
    # 训练深度学习模型
    criterion = nn.CrossEntropyLoss()
    results = []
    
    for model_name in ['TextCNN', 'TextRNN']:
        model = models[model_name]
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环（参考PPT第34页）
        for epoch in range(5):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                features = batch['features']
                labels = batch['label']
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"{model_name} Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        # 验证评估（参考PPT第52页）
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                features = batch['features']
                labels = batch['label']
                
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        results.append({
            'Model': model_name,
            'Accuracy': acc,
            'F1-Score': f1
        })
    
    # 评估朴素贝叶斯
    valid_features = np.array([item['features'].numpy() for item in valid_dataset])
    valid_labels = np.array([item['label'].item() for item in valid_dataset])
    nb_preds = models['NaiveBayes'].predict(valid_features)
    
    acc = accuracy_score(valid_labels, nb_preds)
    f1 = f1_score(valid_labels, nb_preds, average='weighted')
    results.append({
        'Model': 'NaiveBayes',
        'Accuracy': acc,
        'F1-Score': f1
    })
    
    # 输出结果表格（参考PPT第52页）
    print("\n文本分类实验结果:")
    print("+------------+----------+----------+")
    print("|   Model    | Accuracy | F1-Score |")
    print("+------------+----------+----------+")
    for res in results:
        print(f"| {res['Model']:10} | {res['Accuracy']:.4f}  | {res['F1-Score']:.4f}  |")
    print("+------------+----------+----------+")

if __name__ == "__main__":
    main()
  #由于本人没有Python基础，以上代码均由AI完成，本人无法理解上述代码
