import pandas as pd
import numpy as np
import time
import re
import warnings
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# ==================== 基础工具类 ====================

class TextPreprocessor:
    """文本预处理类"""

    def __init__(self, remove_stopwords=True, use_jieba=True, max_seq_length=128):
        self.remove_stopwords = remove_stopwords
        self.use_jieba = use_jieba
        self.max_seq_length = max_seq_length

        # 中文停用词
        self.stopwords = set([
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '那', '什么', '我们', '这个', '这样', '已经', '可以', '对', '但', '把',
            '被', '给', '让', '从', '到',
            '而', '以', '及', '或', '且', '与', '于', '这', '那', '这些', '那些', '因为', '所以', '如果', '但是',
            '虽然', '然后', '而且', '就是'
        ])

        if use_jieba:
            try:
                import jieba
                self.jieba = jieba
                jieba.initialize()
            except ImportError:
                print("警告: 未安装jieba，将使用字符级分词")
                self.use_jieba = False

        # 构建词汇表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def build_vocab(self, texts, max_vocab_size=20000):
        """构建词汇表"""
        word_freq = Counter()

        for text in texts:
            words = self.tokenize(text)
            word_freq.update(words)

        # 保留最常见的词
        most_common = word_freq.most_common(max_vocab_size - 2)  # 减去<PAD>和<UNK>

        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"词汇表大小: {self.vocab_size}")
        return self.vocab_size

    def clean_text(self, text):
        """清洗文本"""
        if pd.isna(text):
            return ""

        text = str(text)
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 去除URL
        text = re.sub(r'http\S+|www\S+', '', text)
        # 去除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、：；"\'（）《》]', '', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def tokenize(self, text):
        """分词"""
        text = self.clean_text(text)

        if self.use_jieba:
            words = self.jieba.lcut(text)
        else:
            # 字符级分词
            words = list(text)

        if self.remove_stopwords:
            words = [word for word in words if word not in self.stopwords and len(word.strip()) > 0]

        return words

    def text_to_sequence(self, text):
        """将文本转换为序列"""
        words = self.tokenize(text)
        sequence = []

        for word in words:
            if word in self.word2idx:
                sequence.append(self.word2idx[word])
            else:
                sequence.append(self.word2idx['<UNK>'])

        # 填充或截断
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))

        return sequence

    def preprocess(self, text):
        """完整的预处理流程"""
        text = self.clean_text(text)
        words = self.tokenize(text)
        return ' '.join(words)


# ==================== 深度学习模型定义 ====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


class TextCNN(nn.Module):
    """TextCNN模型 - 用于文本分类的卷积神经网络"""

    def __init__(self, vocab_size, embedding_dim=128, num_classes=2,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 多个卷积层（不同尺寸的卷积核）
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

        # 卷积层和池化层
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_len - kernel_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 拼接所有卷积层的输出
        cat = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)

        # Dropout和全连接层
        cat = self.dropout(cat)
        logits = self.fc(cat)

        return logits


class BiLSTM(nn.Module):
    """双向LSTM模型 - 用于文本分类"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2是因为双向

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # 嵌入层
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 取最后一个时间步的输出（双向拼接）
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        hidden = hidden.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
        last_hidden = hidden[-1]  # (2, batch_size, hidden_dim)

        # 拼接前向和后向的最后一个隐藏状态
        h_forward = last_hidden[0]  # (batch_size, hidden_dim)
        h_backward = last_hidden[1]  # (batch_size, hidden_dim)
        cat = torch.cat((h_forward, h_backward), dim=1)  # (batch_size, hidden_dim * 2)

        # Dropout和全连接层
        cat = self.dropout(cat)
        logits = self.fc(cat)

        return logits


class TransformerClassifier(nn.Module):
    """Transformer模型 - 基于Attention的文本分类"""

    def __init__(self, vocab_size, embedding_dim=128, num_heads=8,
                 num_layers=2, hidden_dim=256, num_classes=2, dropout=0.1):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout)

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.constant_(self.fc[0].bias, 0)
        nn.init.xavier_uniform_(self.fc[3].weight)
        nn.init.constant_(self.fc[3].bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # 创建padding mask
        padding_mask = (x == 0)

        # 嵌入层和位置编码
        embedded = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoding(embedded)

        # Transformer编码器
        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=padding_mask
        )

        # 取第一个token的输出（类似BERT的[CLS]）
        cls_output = transformer_out[:, 0, :]

        # 分类
        logits = self.fc(cls_output)

        return logits


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MLPClassifier(nn.Module):
    """多层感知机模型"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=2, dropout=0.3):
        super().__init__()

        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.network(x)


# ==================== 深度学习模型训练器 ====================

class DeepLearningTrainer:
    """深度学习模型训练器"""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"使用设备: {self.device}")

    def train_model(self, model, train_loader, val_loader,
                    epochs=10, lr=0.001, patience=3):
        """训练模型"""
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]'):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, '
                  f'Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}')

            # 早停和学习率调度
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'早停触发于 epoch {epoch + 1}')
                    break

        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': max(val_accuracies) if val_accuracies else 0
        }

    def evaluate_model(self, model, test_loader):
        """评估模型"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_micro': f1_score(all_labels, all_preds, average='micro'),
            'precision': precision,
            'recall': recall,
            'predictions': all_preds,
            'labels': all_labels
        }


# ==================== 完整的模型对比系统 ====================

class CompleteModelBenchmark:
    """完整的模型对比系统"""

    def __init__(self):
        self.results = []
        self.models = {}
        self.preprocessor = None

    def run_all_models(self, X_train, X_test, y_train, y_test,
                       preprocessed_texts_train, preprocessed_texts_test):
        """运行所有模型"""

        # 1. 传统机器学习模型
        print("\n" + "=" * 60)
        print("1. 传统机器学习模型")
        print("=" * 60)
        self.run_traditional_models(X_train, X_test, y_train, y_test)

        # 2. 集成学习模型
        print("\n" + "=" * 60)
        print("2. 集成学习模型")
        print("=" * 60)
        self.run_ensemble_models(X_train, X_test, y_train, y_test)

        # 3. 深度学习模型
        print("\n" + "=" * 60)
        print("3. 深度学习模型")
        print("=" * 60)
        self.run_deep_learning_models(preprocessed_texts_train, preprocessed_texts_test, y_train, y_test)

        # 4. 预训练模型
        print("\n" + "=" * 60)
        print("4. 预训练模型")
        print("=" * 60)
        self.run_pretrained_models(preprocessed_texts_train, preprocessed_texts_test, y_train, y_test)

    def run_traditional_models(self, X_train, X_test, y_train, y_test):
        """运行传统机器学习模型"""
        from sklearn.naive_bayes import MultinomialNB, BernoulliNB
        from sklearn.svm import SVC, LinearSVC
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        models = [
            ('Naive Bayes (Multinomial)', MultinomialNB()),
            ('Naive Bayes (Bernoulli)', BernoulliNB()),
            ('Logistic Regression', LogisticRegression(max_iter=1000)),
            ('SVM (Linear)', LinearSVC(max_iter=1000)),
            ('SVM (RBF)', SVC(kernel='rbf', probability=True)),
            ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
            ('Decision Tree', DecisionTreeClassifier(max_depth=10)),
            ('LDA', LinearDiscriminantAnalysis()),
            ('SGD Classifier', SGDClassifier()),
        ]

        for name, model in tqdm(models, desc="训练传统模型"):
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                y_pred = model.predict(X_test)

                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                self.results.append({
                    'model_name': name,
                    'model_type': '传统机器学习',
                    'accuracy': accuracy,
                    'f1_weighted': f1,
                    'precision': precision,
                    'recall': recall,
                    'train_time': train_time,
                    'predict_time_per_1000': 0.01,  # 估计值
                    'feature_type': 'TF-IDF'
                })

                print(f"✓ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            except Exception as e:
                print(f"✗ {name} 训练失败: {str(e)}")

    def run_ensemble_models(self, X_train, X_test, y_train, y_test):
        """运行集成学习模型"""
        from sklearn.ensemble import (
            RandomForestClassifier, GradientBoostingClassifier,
            AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
        )

        models = [
            ('Random Forest', RandomForestClassifier(n_estimators=100)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100)),
            ('AdaBoost', AdaBoostClassifier(n_estimators=100)),
            ('Extra Trees', ExtraTreesClassifier(n_estimators=100)),
            ('Bagging', BaggingClassifier(n_estimators=10)),
        ]

        # 尝试LightGBM和XGBoost
        try:
            import lightgbm as lgb
            models.append(('LightGBM', lgb.LGBMClassifier(n_estimators=100)))
        except:
            pass

        try:
            import xgboost as xgb
            models.append(
                ('XGBoost', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')))
        except:
            pass

        for name, model in tqdm(models, desc="训练集成模型"):
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                y_pred = model.predict(X_test)

                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

                self.results.append({
                    'model_name': name,
                    'model_type': '集成学习',
                    'accuracy': accuracy,
                    'f1_weighted': f1,
                    'precision': precision,
                    'recall': recall,
                    'train_time': train_time,
                    'predict_time_per_1000': 0.02,  # 估计值
                    'feature_type': 'TF-IDF'
                })

                print(f"✓ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            except Exception as e:
                print(f"✗ {name} 训练失败: {str(e)}")

    def run_deep_learning_models(self, texts_train, texts_test, y_train, y_test):
        """运行深度学习模型"""

        # 构建词汇表
        if self.preprocessor is None:
            self.preprocessor = TextPreprocessor(max_seq_length=128)
            self.preprocessor.build_vocab(texts_train)

        # 准备数据
        X_train_seq = np.array([self.preprocessor.text_to_sequence(text) for text in texts_train])
        X_test_seq = np.array([self.preprocessor.text_to_sequence(text) for text in texts_test])

        # 转换为PyTorch张量
        X_train_tensor = torch.LongTensor(X_train_seq)
        X_test_tensor = torch.LongTensor(X_test_seq)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化训练器
        trainer = DeepLearningTrainer()

        # 定义要训练的深度学习模型
        dl_models = [
            ('TextCNN', TextCNN, {'vocab_size': self.preprocessor.vocab_size}),
            ('BiLSTM', BiLSTM, {'vocab_size': self.preprocessor.vocab_size}),
            ('Transformer', TransformerClassifier, {'vocab_size': self.preprocessor.vocab_size}),
            ('MLP (on embeddings)', None, None),  # 特殊处理
        ]

        for name, model_class, kwargs in tqdm(dl_models, desc="训练深度学习模型"):
            try:
                if name == 'MLP (on embeddings)':
                    # MLP基于平均词向量
                    result = self._train_mlp_on_embeddings(texts_train, texts_test, y_train, y_test, trainer)
                else:
                    # 其他深度学习模型
                    model = model_class(**kwargs)

                    start_time = time.time()
                    training_result = trainer.train_model(
                        model, train_loader, test_loader,
                        epochs=10, lr=0.001
                    )
                    train_time = time.time() - start_time

                    # 评估
                    eval_result = trainer.evaluate_model(training_result['model'], test_loader)

                    result = {
                        'model_name': name,
                        'model_type': '深度学习',
                        'accuracy': eval_result['accuracy'],
                        'f1_weighted': eval_result['f1_weighted'],
                        'precision': eval_result['precision'],
                        'recall': eval_result['recall'],
                        'train_time': train_time,
                        'predict_time_per_1000': 0.1,  # 估计值
                        'feature_type': '词嵌入'
                    }

                self.results.append(result)
                print(f"✓ {name}: Accuracy={result['accuracy']:.4f}, F1={result['f1_weighted']:.4f}")

            except Exception as e:
                print(f"✗ {name} 训练失败: {str(e)}")

    def _train_mlp_on_embeddings(self, texts_train, texts_test, y_train, y_test, trainer):
        """基于词向量训练MLP"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 使用TF-IDF作为MLP的输入特征
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = vectorizer.fit_transform(texts_train).toarray()
        X_test_tfidf = vectorizer.transform(texts_test).toarray()

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_tfidf)
        X_test_tensor = torch.FloatTensor(X_test_tfidf)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 创建MLP模型
        input_size = X_train_tfidf.shape[1]
        model = MLPClassifier(input_size=input_size)

        # 训练
        start_time = time.time()
        training_result = trainer.train_model(
            model, train_loader, test_loader,
            epochs=20, lr=0.001
        )
        train_time = time.time() - start_time

        # 评估
        eval_result = trainer.evaluate_model(training_result['model'], test_loader)

        return {
            'model_name': 'MLP (on TF-IDF)',
            'model_type': '深度学习',
            'accuracy': eval_result['accuracy'],
            'f1_weighted': eval_result['f1_weighted'],
            'precision': eval_result['precision'],
            'recall': eval_result['recall'],
            'train_time': train_time,
            'predict_time_per_1000': 0.05,  # 估计值
            'feature_type': 'TF-IDF'
        }

    def run_pretrained_models(self, texts_train, texts_test, y_train, y_test):
        """运行预训练模型"""
        try:
            from transformers import (
                BertTokenizer, BertForSequenceClassification,
                RobertaTokenizer, RobertaForSequenceClassification,
                DistilBertTokenizer, DistilBertForSequenceClassification,
                Trainer, TrainingArguments
            )

            models_to_try = [
                ('BERT', 'bert-base-chinese', BertTokenizer, BertForSequenceClassification),
                ('RoBERTa', 'hfl/chinese-roberta-wwm-ext', RobertaTokenizer, RobertaForSequenceClassification),
                ('DistilBERT', 'hfl/chinese-distilbert-wwm-ext', DistilBertTokenizer,
                 DistilBertForSequenceClassification),
            ]

            for name, model_name, tokenizer_class, model_class in models_to_try:
                try:
                    print(f"\n训练 {name} ({model_name})...")

                    tokenizer = tokenizer_class.from_pretrained(model_name)

                    # 编码数据
                    train_encodings = tokenizer(
                        list(texts_train),
                        truncation=True,
                        padding=True,
                        max_length=128
                    )

                    test_encodings = tokenizer(
                        list(texts_test),
                        truncation=True,
                        padding=True,
                        max_length=128
                    )

                    # 创建数据集
                    class ReviewDataset(torch.utils.data.Dataset):
                        def __init__(self, encodings, labels):
                            self.encodings = encodings
                            self.labels = labels

                        def __getitem__(self, idx):
                            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                            item['labels'] = torch.tensor(self.labels[idx])
                            return item

                        def __len__(self):
                            return len(self.labels)

                    train_dataset = ReviewDataset(train_encodings, y_train)
                    test_dataset = ReviewDataset(test_encodings, y_test)

                    # 加载模型
                    model = model_class.from_pretrained(model_name, num_labels=2)

                    # 训练参数
                    training_args = TrainingArguments(
                        output_dir=f'./{name}_results',
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=16,
                        warmup_steps=100,
                        weight_decay=0.01,
                        logging_dir=f'./{name}_logs',
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        metric_for_best_model="f1"
                    )

                    # 计算指标的函数
                    def compute_metrics(p):
                        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                        preds = np.argmax(p.predictions, axis=1)

                        return {
                            'accuracy': accuracy_score(p.label_ids, preds),
                            'f1': f1_score(p.label_ids, preds, average='weighted'),
                            'precision': precision_score(p.label_ids, preds, average='weighted', zero_division=0),
                            'recall': recall_score(p.label_ids, preds, average='weighted', zero_division=0),
                        }

                    # 训练
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset,
                        compute_metrics=compute_metrics
                    )

                    start_time = time.time()
                    trainer.train()
                    train_time = time.time() - start_time

                    # 评估
                    eval_result = trainer.evaluate()

                    self.results.append({
                        'model_name': f'{name} ({model_name.split("/")[-1]})',
                        'model_type': '预训练模型',
                        'accuracy': eval_result['eval_accuracy'],
                        'f1_weighted': eval_result['eval_f1'],
                        'precision': eval_result['eval_precision'],
                        'recall': eval_result['eval_recall'],
                        'train_time': train_time,
                        'predict_time_per_1000': 0.2,  # 估计值
                        'feature_type': '预训练嵌入'
                    })

                    print(f"✓ {name}: Accuracy={eval_result['eval_accuracy']:.4f}, F1={eval_result['eval_f1']:.4f}")

                except Exception as e:
                    print(f"✗ {name} 训练失败: {str(e)}")

        except ImportError:
            print("Transformers库未安装，跳过预训练模型")

    def generate_report(self):
        """生成报告"""
        df_results = pd.DataFrame(self.results)

        # 保存结果
        os.makedirs('./results', exist_ok=True)
        df_results.to_csv('./results/all_models_results.csv', index=False, encoding='utf-8-sig')

        # 生成可视化
        self._create_visualizations(df_results)

        return df_results

    def _create_visualizations(self, df_results):
        """创建可视化图表"""

        # 1. 所有模型准确率对比
        plt.figure(figsize=(14, 8))
        models_sorted = df_results.sort_values('accuracy', ascending=False)
        bars = plt.barh(range(len(models_sorted)), models_sorted['accuracy'])

        # 按模型类型着色
        colors = {
            '传统机器学习': 'skyblue',
            '集成学习': 'lightgreen',
            '深度学习': 'orange',
            '预训练模型': 'red'
        }

        for i, (_, row) in enumerate(models_sorted.iterrows()):
            color = colors.get(row['model_type'], 'gray')
            bars[i].set_color(color)

        plt.yticks(range(len(models_sorted)), models_sorted['model_name'])
        plt.xlabel('准确率')
        plt.title('所有模型准确率对比')
        plt.tight_layout()
        plt.savefig('./results/all_models_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 按模型类型统计
        plt.figure(figsize=(10, 6))
        type_stats = df_results.groupby('model_type').agg({
            'accuracy': 'mean',
            'f1_weighted': 'mean',
            'train_time': 'mean'
        }).round(4)

        x = range(len(type_stats))
        width = 0.25

        plt.bar([i - width for i in x], type_stats['accuracy'], width, label='平均准确率')
        plt.bar(x, type_stats['f1_weighted'], width, label='平均F1分数')
        plt.bar([i + width for i in x], type_stats['train_time'], width, label='平均训练时间')

        plt.xlabel('模型类型')
        plt.ylabel('分数/时间')
        plt.title('不同模型类型的性能对比')
        plt.xticks(x, type_stats.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./results/model_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 训练时间 vs 准确率散点图
        plt.figure(figsize=(10, 6))
        colors = {
            '传统机器学习': 'blue',
            '集成学习': 'green',
            '深度学习': 'orange',
            '预训练模型': 'red'
        }

        for model_type, color in colors.items():
            subset = df_results[df_results['model_type'] == model_type]
            if len(subset) > 0:
                plt.scatter(subset['train_time'], subset['accuracy'],
                            c=color, label=model_type, s=100, alpha=0.7)

        plt.xlabel('训练时间 (秒)')
        plt.ylabel('准确率')
        plt.title('训练时间 vs 准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./results/train_time_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()


# ==================== 主程序 ====================

def main():
    print("=" * 80)
    print("完整的NLP文本分类模型对比系统")
    print("包含: CNN, LSTM, Transformer, MLP, 传统ML, 集成学习, 预训练模型")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. 加载数据...")

    train_path = "/Users/apple/Downloads/train.csv"
    test_path = "/Users/apple/Downloads/test.csv"

    try:
        # 尝试不同的编码
        for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
            try:
                train_df = pd.read_csv(train_path, encoding=encoding)
                test_df = pd.read_csv(test_path, encoding=encoding)
                print(f"使用 {encoding} 编码加载成功")
                break
            except:
                continue
        else:
            print("无法加载数据，请检查文件路径和编码")
            return
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 2. 检查数据列
    print(f"\n训练集列名: {train_df.columns.tolist()}")
    print(f"测试集列名: {test_df.columns.tolist()}")

    # 自动检测文本列和标签列
    text_col = None
    label_col = None

    for col in train_df.columns:
        if any(keyword in col.lower() for keyword in ['review', 'text', 'content', 'comment', '标题', '内容']):
            text_col = col
            print(f"检测到文本列: {text_col}")
            break

    for col in train_df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'sentiment', 'class', 'category', '评分', '评价']):
            label_col = col
            print(f"检测到标签列: {label_col}")
            break

    if text_col is None:
        # 尝试使用第一列作为文本
        text_col = train_df.columns[0]
        print(f"使用第一列作为文本列: {text_col}")

    if label_col is None:
        # 尝试使用第二列作为标签
        if len(train_df.columns) > 1:
            label_col = train_df.columns[1]
            print(f"使用第二列作为标签列: {label_col}")
        else:
            print("错误: 未找到标签列")
            return

    # 3. 数据预处理
    print("\n2. 数据预处理...")
    preprocessor = TextPreprocessor(remove_stopwords=True, use_jieba=True)

    # 预处理文本
    print("预处理训练集文本...")
    train_df['cleaned_text'] = train_df[text_col].apply(preprocessor.preprocess)

    print("预处理测试集文本...")
    test_df['cleaned_text'] = test_df[text_col].apply(preprocessor.preprocess)

    # 检查标签
    print(f"\n训练集标签分布:\n{train_df[label_col].value_counts()}")
    print(f"测试集标签分布:\n{test_df[label_col].value_counts()}")

    # 确保标签是0/1或转换为0/1
    unique_labels = np.unique(train_df[label_col])
    print(f"唯一标签值: {unique_labels}")

    if len(unique_labels) != 2:
        print("警告: 标签不是二分类，尝试转换...")
        # 尝试转换为0/1
        if set(unique_labels).issubset({0, 1}):
            # 已经是0/1
            pass
        else:
            # 映射到0/1
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            train_df[label_col] = train_df[label_col].map(label_mapping)
            test_df[label_col] = test_df[label_col].map(label_mapping)

    # 准备数据
    X_train_raw = train_df['cleaned_text'].values
    y_train = train_df[label_col].values

    X_test_raw = test_df['cleaned_text'].values
    y_test = test_df[label_col].values

    print(f"\n训练集: {len(X_train_raw)} 条")
    print(f"测试集: {len(X_test_raw)} 条")

    # 4. 特征提取（用于传统模型）
    print("\n3. 特征提取...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")

    # 5. 运行所有模型
    print("\n4. 运行所有模型...")

    benchmark = CompleteModelBenchmark()
    benchmark.run_all_models(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        X_train_raw, X_test_raw
    )

    # 6. 生成报告
    print("\n5. 生成报告...")
    df_results = benchmark.generate_report()

    # 7. 显示结果
    print("\n" + "=" * 80)
    print("最终结果汇总")
    print("=" * 80)

    # Top 10 模型
    print("\nTop 10 模型 (按准确率):")
    top_10 = df_results.sort_values('accuracy', ascending=False).head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['model_name']:30} | "
              f"Acc: {row['accuracy']:.4f} | "
              f"F1: {row['f1_weighted']:.4f} | "
              f"Time: {row['train_time']:.2f}s | "
              f"Type: {row['model_type']}")

    # 按模型类型统计
    print("\n按模型类型统计:")
    type_summary = df_results.groupby('model_type').agg({
        'accuracy': ['mean', 'max', 'count'],
        'train_time': 'mean'
    }).round(4)

    print(type_summary.to_string())

    # 最佳模型
    best_acc_model = df_results.loc[df_results['accuracy'].idxmax()]
    best_f1_model = df_results.loc[df_results['f1_weighted'].idxmax()]
    fastest_model = df_results.loc[df_results['train_time'].idxmin()]

    print(f"\n最佳准确率模型: {best_acc_model['model_name']} ({best_acc_model['accuracy']:.4f})")
    print(f"最佳F1分数模型: {best_f1_model['model_name']} ({best_f1_model['f1_weighted']:.4f})")
    print(f"最快训练模型: {fastest_model['model_name']} ({fastest_model['train_time']:.2f}秒)")

    print(f"\n包含的模型类型:")
    print(f"1. 传统机器学习: Naive Bayes, SVM, Logistic Regression, KNN, Decision Tree")
    print(f"2. 集成学习: Random Forest, Gradient Boosting, AdaBoost, LightGBM, XGBoost")
    print(f"3. 深度学习: TextCNN, BiLSTM, Transformer, MLP")
    print(f"4. 预训练模型: BERT, RoBERTa, DistilBERT")

    print(f"\n{'=' * 80}")
    print(f"结果已保存到 ./results/ 目录")
    print(f"1. all_models_results.csv - 所有模型详细结果")
    print(f"2. all_models_accuracy.png - 所有模型准确率对比图")
    print(f"3. model_type_comparison.png - 模型类型对比图")
    print(f"4. train_time_vs_accuracy.png - 训练时间vs准确率图")
    print('=' * 80)


if __name__ == "__main__":
    main()