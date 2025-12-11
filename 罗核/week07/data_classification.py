
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from torch.utils.data import DataLoader
from data_loader import CSVDataLoader
from models import TextCNN, LSTMClassifier, TransformerClassifier, MLPClassifier
from trainer import TextClassificationTrainer, evaluate_model
import matplotlib.pyplot as plt
import os

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 文本预处理和编码
        tokens = self.preprocess_text(text)
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # 填充或截断
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def preprocess_text(self, text):
        # 转换为小写
        text = text.lower()
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 分词
        tokens = text.split()
        return tokens

class CSVDataLoader:
    def __init__(self, csv_path, text_column='text', label_column='label'):
        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_size = 2
    
    def load_data(self):
        """加载CSV数据"""
        df = pd.read_csv(self.csv_path)
        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()
        return texts, labels
    
    def build_vocab(self, texts, max_vocab_size=10000):
        """构建词汇表"""
        word_counts = Counter()
        
        for text in texts:
            tokens = self.preprocess_text(text)
            word_counts.update(tokens)
        
        # 添加最常见的词
        for word, count in word_counts.most_common(max_vocab_size):
            if word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1
        
        return self.vocab
    
    def preprocess_text(self, text):
        # 转换为小写
        text = text.lower()
        # 移除特殊字符和数字
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 分词
        tokens = text.split()
        return tokens
    
    def create_data_loaders(self, texts, labels, batch_size=32, max_length=100):
        """创建数据加载器"""
        # 划分训练集和测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 构建词汇表
        vocab = self.build_vocab(texts)
        
        # 创建数据集
        train_dataset = TextDataset(train_texts, train_labels, vocab, max_length)
        test_dataset = TextDataset(test_texts, test_labels, vocab, max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, vocab

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, kernel_sizes=[3,4,5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size, padding=(kernel_size-1)//2)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                        batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return output

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_length=100):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        embedded = embedded.permute(1, 0, 2)
        transformer_out = self.transformer_encoder(embedded)
        output = self.fc(transformer_out[0])
        return output

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dims, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers = []
        prev_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        mlp_out = self.mlp(pooled)
        output = self.fc(mlp_out)
        return output

class TextClassificationTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.train_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def train(self, epochs=10):
        print("开始训练...")
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_accuracy = self.validate()
            self.scheduler.step()
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}, '
                  f'Time: {epoch_time:.2f}s')
        
        return self.train_losses, self.val_accuracies

def evaluate_model(model, test_loader, device, label_names):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"测试准确率: {accuracy:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_predictions, target_names=label_names))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    return accuracy

def create_sample_csv():
    """创建示例CSV文件用于测试"""
    data = {
        'text': [
            'football match ends in dramatic victory for the home team',
            'new smartphone features advanced camera technology and AI',
            'stock market shows positive growth trend this quarter',
            'basketball team advances to finals after close game',
            'tech company releases latest software update with new features',
            'company reports strong quarterly earnings exceeding expectations',
            'tennis player wins championship title in straight sets',
            'artificial intelligence startup raises significant funding round'
        ],
        'label': [0, 1, 2, 0, 1, 2, 0, 1]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_dataset.csv', index=False)
    print("示例CSV文件已创建: sample_dataset.csv")
    return 'sample_dataset.csv'

def main():
    print("=" * 60)
    print("     PyTorch CSV文本分类系统")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建示例CSV文件
    csv_path = create_sample_csv()
    
    # 初始化数据加载器
    data_loader = CSVDataLoader(csv_path, text_column='text', label_column='label')
    
    # 加载数据
    print("\n1. 加载CSV数据...")
    texts, labels = data_loader.load_data()
    print(f"数据集大小: {len(texts)}")
    
    # 创建数据加载器
    train_loader, test_loader, vocab = data_loader.create_data_loaders(texts, labels)
    
    # 模型配置
    config = {
        'vocab_size': len(vocab),
        'embed_dim': 100,
        'num_classes': 3  # 假设有3个类别
    }
    
    # 定义要训练的模型
    models = {
        'TextCNN': TextCNN(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_classes=config['num_classes']
        ),
        'LSTM': LSTMClassifier(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dim=128,
            num_classes=config['num_classes']
        ),
        'Transformer': TransformerClassifier(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=4,
            hidden_dim=256,
            num_layers=3,
            num_classes=config['num_classes']
        ),
        'MLP': MLPClassifier(
            vocab_size=config['vocab_size'],
            embed_dim=config['embed_dim'],
            hidden_dims=[256, 128],
            num_classes=config['num_classes']
        )
    }
    
    # 训练和评估每个模型
    results = {}
    label_names = ['体育', '科技', '财经']
    
    for model_name, model in models.items():
        print(f"\n2. 训练 {model_name} 模型...")
        
        trainer = TextClassificationTrainer(model, train_loader, test_loader, device)
        train_losses, val_accuracies = trainer.train(epochs=5)
        
        # 评估模型
        accuracy = evaluate_model(model, test_loader, device, label_names)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
    
    # 比较模型性能
    print("\n3. 模型性能比较...")
    model_names = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('模型准确率比较')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.show()
    
    # 保存最佳模型
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    torch.save(results[best_model_name]['model'].state_dict(), f'best_{best_model_name}_model.pth')
    print(f"\n最佳模型已保存: best_{best_model_name}_model.pth")
    
    print("\n" + "=" * 60)
    print("     CSV文本分类训练完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()