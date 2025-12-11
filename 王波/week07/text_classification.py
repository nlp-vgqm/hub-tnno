import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
import re
import warnings

warnings.filterwarnings('ignore')

# 文本处理
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import logging

jieba.setLogLevel(logging.ERROR)

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# 评估和工具
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os


class DataAnalyzer:
    """数据分析器"""
    def __init__(self, data_path=None, data=None, text_col='review', label_col='label'):
        self.df = pd.read_csv(data_path, encoding='utf-8')
        self.text_col = text_col
        self.label_col = label_col
        self.analyze()

    def analyze(self):
        """进行数据分析"""
        print("=" * 50)
        print("数据统计分析")
        print("=" * 50)

        # 基本统计
        print(f"总样本数: {len(self.df)}")

        # 统计好评和差评数量
        if '好评' in self.df[self.label_col].values and '差评' in self.df[self.label_col].values:
            print(f"正样本数(好评): {len(self.df[self.df[self.label_col] == '好评'])}")
            print(f"负样本数(差评): {len(self.df[self.df[self.label_col] == '差评'])}")
        else:
            # 显示所有标签的分布
            label_counts = self.df[self.label_col].value_counts()
            print("标签分布:")
            for label, count in label_counts.items():
                print(f"  {label}: {count}")

        # 文本长度分析
        self.df['text_length'] = self.df[self.text_col].apply(len)
        self.df['word_count'] = self.df[self.text_col].apply(lambda x: len(jieba.lcut(x)))

        print(f"\n文本长度统计:")
        print(f"平均字符数: {self.df['text_length'].mean():.2f}")
        print(f"最大字符数: {self.df['text_length'].max()}")
        print(f"最小字符数: {self.df['text_length'].min()}")

        print(f"\n分词后统计:")
        print(f"平均词数: {self.df['word_count'].mean():.2f}")

        # 词频分析
        self.analyze_word_freq()

        return self.df

    def analyze_word_freq(self):
        """分析词频"""
        all_text = ' '.join(self.df[self.text_col].tolist())
        words = jieba.lcut(all_text)

        # 过滤停用词和短词
        stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就',
                          '不', '人', '都', '一', '一个', '上', '也', '很',
                          '到', '说', '要', '去', '你', '会', '着', '没有',
                          '看', '好', '自己', '这'])

        words_filtered = [w for w in words if len(w) > 1 and w not in stop_words]

        # 统计词频
        word_freq = Counter(words_filtered)

        print(f"\n高频词汇(top 10):")
        for word, freq in word_freq.most_common(10):
            print(f"  {word}: {freq}")

    def visualize(self):
        """可视化分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 类别分布
        ax1 = axes[0, 0]
        label_counts = self.df[self.label_col].value_counts()
        colors = ['green', 'red'] if len(label_counts) == 2 else plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
        ax1.bar(label_counts.index, label_counts.values, color=colors)
        ax1.set_title('类别分布')
        ax1.set_ylabel('数量')
        ax1.tick_params(axis='x', rotation=45)

        # 文本长度分布
        ax2 = axes[0, 1]
        ax2.hist(self.df['text_length'], bins=30, alpha=0.7, color='blue')
        ax2.set_title('文本长度分布')
        ax2.set_xlabel('字符数')
        ax2.set_ylabel('频数')

        # 词数分布
        ax3 = axes[1, 0]
        ax3.hist(self.df['word_count'], bins=30, alpha=0.7, color='orange')
        ax3.set_title('词数分布')
        ax3.set_xlabel('词数')
        ax3.set_ylabel('频数')

        # 箱线图
        ax4 = axes[1, 1]
        data_to_plot = []
        labels = []
        for label in self.df[self.label_col].unique():
            data_to_plot.append(self.df[self.df[self.label_col] == label]['text_length'])
            labels.append(label)
        ax4.boxplot(data_to_plot, labels=labels)
        ax4.set_title('文本长度箱线图')
        ax4.set_ylabel('字符数')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self):
        self.vectorizer = None

    def clean_text(self, text):
        """清洗文本"""
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 保留中文和字母数字
        text = re.sub(r'\d+', '', text)  # 去除数字
        text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空格
        return text

    def tokenize(self, text):
        """分词"""
        return jieba.lcut(text)

    def preprocess(self, texts, fit=True):
        """预处理文本"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        # ' '（一个空格字符串）是连接符，它的作用是将分词后的词语列表用空格重新连接成一个字符串
        # 为什么用空格？
        # 因为TF-IDF向量化器默认按空格分隔词语
        tokenized_texts = [' '.join(self.tokenize(text)) for text in cleaned_texts]
        return tokenized_texts

    def extract_tfidf_features(self, texts, fit=True, max_features=3000):
        """提取TF-IDF特征"""
        if fit or self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                token_pattern=r'(?u)\b\w+\b',
                ngram_range=(1, 2)  # 使用unigram和bigram
            )
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)
        return features


class ModelComparator:
    """模型比较器"""

    def __init__(self):
        self.results = []

    def train_and_evaluate(self, model_info, X_train, X_val, y_train, y_val):
        """训练和评估单个模型"""
        model_name = model_info['name']
        model = model_info['model']
        learning_rate = model_info.get('learning_rate', None)
        hidden_size = model_info.get('hidden_size', None)

        print(f"\n训练 {model_name}...")
        if learning_rate is not None:
            print(f"  学习率: {learning_rate}")
        if hidden_size is not None:
            print(f"  隐藏层大小: {hidden_size}")

        # 训练时间
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

        # 预测时间
        start_predict = time.time()
        y_pred = model.predict(X_val)
        predict_time = time.time() - start_predict

        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        # F1分数是机器学习和统计分类问题中常用的一个衡量模型性能的指标，它同时考虑了精确率（Precision）和召回率（Recall），是二者的调和平均数。
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        # 获取样本数量（使用shape[0]而不是len()）
        n_samples = X_val.shape[0]

        # 保存结果
        result = {
            '模型': model_name,
            '准确率(acc)': accuracy,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1,
            '训练时间(s)': train_time,
            '预测时间(s)': predict_time,
            '预测时间/样本(ms)': (predict_time / n_samples) * 1000 if n_samples > 0 else 0,
            '学习率': learning_rate if learning_rate is not None else '-',
            '隐藏层大小': hidden_size if hidden_size is not None else '-'
        }

        self.results.append(result)

        print(f"  准确率(acc): {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  训练时间: {train_time:.3f}s")
        print(f"  预测时间: {predict_time:.3f}s")

        return result


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, data_path=None, data=None):
        self.data_path = data_path
        self.data = data
        self.comparator = ModelComparator()
        self.models = []
        self.feature_names = []

    def setup_models(self):
        """设置要比较的模型"""
        # 基础模型配置（传统机器学习模型）
        self.models = [
            {
                'name': '逻辑回归',
                'model': LogisticRegression(max_iter=2000, random_state=42, C=0.8),
                'learning_rate': None,
                'hidden_size': None
            },
            {
                'name': '朴素贝叶斯',
                'model': MultinomialNB(alpha=1.0),
                'learning_rate': None,
                'hidden_size': None
            },
            {
                'name': '支持向量机',
                'model': SVC(kernel='linear', probability=True, random_state=42, C=1.0),
                'learning_rate': None,
                'hidden_size': None
            },
            {
                'name': '随机森林',
                'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None),
                'learning_rate': None,
                'hidden_size': None
            },
            # MLP神经网络模型 - 不同学习率
            {
                'name': 'MLP(学习率0.001)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(100,),
                    learning_rate_init=0.001,
                    max_iter=500,    # 最大迭代次数，即最大 epoch 数
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.001,
                'hidden_size': 100
            },
            {
                'name': 'MLP(学习率0.01)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(100,),
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.01,
                'hidden_size': 100
            },
            # MLP神经网络模型 - 不同隐藏层大小
            {
                'name': 'MLP(隐藏层50)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(50,),
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.01,
                'hidden_size': 50
            },
            {
                'name': 'MLP(隐藏层100,50)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.01,
                'hidden_size': '100,50'
            },
            {
                'name': 'MLP(隐藏层200,100)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(200, 100),
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.01,
                'hidden_size': '200,100'
            },
            {
                'name': 'MLP(隐藏层400,200,200,学习率0.01)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(400, 200, 200),
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'learning_rate': 0.01,
                'hidden_size': '400,200,200'
            }
        ]

        # 尝试添加额外的模型
        try:
            from sklearn.linear_model import SGDClassifier
            self.models.append({
                'name': 'SGD分类器',   # 随机梯度下降
                'model': SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    max_iter=1000,
                    random_state=42,
                    learning_rate='optimal'
                ),
                'learning_rate': 'optimal',
                'hidden_size': None
            })
        except:
            pass

        try:
            from sklearn.neighbors import KNeighborsClassifier
            self.models.append({
                'name': 'KNN',    # k近邻算法
                'model': KNeighborsClassifier(n_neighbors=5),
                'learning_rate': None,
                'hidden_size': None
            })
        except:
            pass

    def run_experiment(self):
        """运行完整实验"""
        print("=" * 60)
        print("网络购物评论分类实验")
        print("=" * 60)

        # 1. 数据加载和分析
        print("\n1. 数据加载和分析")
        analyzer = DataAnalyzer(data_path=self.data_path, data=self.data)
        df = analyzer.df

        # 2. 数据预处理
        print("\n2. 数据预处理")
        preprocessor = TextPreprocessor()
        texts = df['review'].tolist()
        labels = df['label'].tolist()

        # 编码标签，此步骤可以省略
        le = LabelEncoder()
        y = le.fit_transform(labels)

        # 显示标签映射
        print(f"标签编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # 预处理文本
        processed_texts = preprocessor.preprocess(texts)

        # 提取特征
        X = preprocessor.extract_tfidf_features(processed_texts, fit=True)
        self.feature_names = preprocessor.vectorizer.get_feature_names_out()

        print(f"特征维度: {X.shape}")
        print(f"特征数量: {len(self.feature_names)}")

        # 3. 划分训练集和验证集
        # train_test_split(
        # X‌：特征数据（输入变量），通常为二维数组或 DataFrame。
        # y‌：目标变量（输出变量），通常为一维数组或 Series。
        # test_size‌：测试集占比（浮点数，如 0.25 表示 25%）或样本数量（整数）。‌
        # random_state‌：随机种子，确保每次拆分结果可重复（例如，设置为 0 或 1 时结果一致）
        # shuffle‌：布尔值，是否在拆分前打乱数据（默认为 True）。‌
        # tratify‌：如果指定，会按目标变量的分布进行分层抽样，确保训练集和测试集中的类别比例一致（适用于分类问题）)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n训练集大小: {X_train.shape[0]}")  # 使用shape[0]获取样本数
        print(f"验证集大小: {X_val.shape[0]}")  # 使用shape[0]获取样本数

        # 4. 设置模型
        self.setup_models()

        # 5. 训练和评估模型
        print("\n3. 模型训练和评估")
        print("-" * 60)

        for model_info in self.models:
            try:
                self.comparator.train_and_evaluate(
                    model_info, X_train, X_val, y_train, y_val
                )
            except Exception as e:
                print(f"训练 {model_info['name']} 时出错: {e}")

        # 6. 显示结果
        print("\n4. 实验结果总结")
        print("=" * 60)

        results_df = pd.DataFrame(self.comparator.results)
        results_df = results_df.sort_values('准确率(acc)', ascending=False)

        self.display_results_table(results_df)
        self.visualize_results(results_df)
        self.analyze_model_parameters(results_df)
        self.analyze_important_features(results_df, preprocessor.vectorizer)

        return results_df

    def display_results_table(self, results_df):
        """显示结果表格"""
        print("\n模型性能对比表:")
        print("-" * 100)
        print(
            f"{'模型':<20} {'准确率(acc)':<12} {'F1分数':<10} {'学习率':<10} {'隐藏层大小':<12} {'训练时间(s)':<12} {'预测时间/样本(ms)':<20}")
        print("-" * 100)

        for _, row in results_df.iterrows():
            print(f"{row['模型']:<20} {row['准确率(acc)']:<12.4f} {row['F1分数']:<10.4f} "
                  f"{str(row['学习率']):<10} {str(row['隐藏层大小']):<12} "
                  f"{row['训练时间(s)']:<12.3f} {row['预测时间/样本(ms)']:<20.4f}")

        # 保存结果到文件
        results_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8')
        print(f"\n详细结果已保存到: model_comparison_results.csv")

    def visualize_results(self, results_df):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 准确率对比
        ax1 = axes[0, 0]
        models = results_df['模型']
        accuracy = results_df['准确率(acc)']
        bars1 = ax1.barh(models, accuracy, color='skyblue')
        ax1.set_xlabel('准确率(acc)')
        ax1.set_title('模型准确率对比')
        ax1.set_xlim([0, 1])
        # 在条形上添加数值
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)

        # 2. F1分数对比
        ax2 = axes[0, 1]
        f1_scores = results_df['F1分数']
        bars2 = ax2.barh(models, f1_scores, color='lightgreen')
        ax2.set_xlabel('F1分数')
        ax2.set_title('模型F1分数对比')
        ax2.set_xlim([0, 1])
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)

        # 3. 训练时间对比
        ax3 = axes[0, 2]
        train_times = results_df['训练时间(s)']
        bars3 = ax3.barh(models, train_times, color='salmon')
        ax3.set_xlabel('训练时间(s)')
        ax3.set_title('模型训练时间对比')
        for bar in bars3:
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}s', ha='left', va='center', fontsize=8)

        # 4. 预测速度对比
        ax4 = axes[1, 0]
        pred_times = results_df['预测时间/样本(ms)']
        bars4 = ax4.barh(models, pred_times, color='gold')
        ax4.set_xlabel('预测时间/样本(ms)')
        ax4.set_title('模型预测速度对比')
        for bar in bars4:
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height() / 2,
                     f'{width:.2f}ms', ha='left', va='center', fontsize=8)

        # 5. 学习率与准确率关系（仅对有学习率的模型）
        ax5 = axes[1, 1]
        mlp_results = results_df[results_df['学习率'] != '-']
        if len(mlp_results) > 0:
            # 提取数值型学习率
            learning_rates = []
            accuracies = []
            for _, row in mlp_results.iterrows():
                try:
                    lr = float(row['学习率'])
                    learning_rates.append(lr)
                    accuracies.append(row['准确率(acc)'])
                except:
                    pass

            if len(learning_rates) > 0:
                ax5.scatter(learning_rates, accuracies, color='purple', s=100)
                ax5.set_xlabel('学习率')
                ax5.set_ylabel('准确率(acc)')
                ax5.set_title('学习率 vs 准确率')
                ax5.set_xscale('log')
                # 添加模型标签
                for i, (_, row) in enumerate(mlp_results.iterrows()):
                    try:
                        lr = float(row['学习率'])
                        ax5.annotate(row['模型'], (lr, row['准确率(acc)']),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)
                    except:
                        pass

        # 6. 隐藏层大小与性能关系
        ax6 = axes[1, 2]
        # 只显示有隐藏层大小的模型
        hidden_size_results = results_df[results_df['隐藏层大小'] != '-']
        if len(hidden_size_results) > 0:
            hidden_sizes = []
            accuracies = []
            for _, row in hidden_size_results.iterrows():
                hidden_sizes.append(str(row['隐藏层大小']))
                accuracies.append(row['准确率(acc)'])

            # 创建条形图
            x_pos = np.arange(len(hidden_sizes))
            bars6 = ax6.bar(x_pos, accuracies, color='lightcoral')
            ax6.set_xlabel('隐藏层大小')
            ax6.set_ylabel('准确率(acc)')
            ax6.set_title('隐藏层大小 vs 准确率')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(hidden_sizes, rotation=45, ha='right')
            # 添加数值标签
            for bar, acc in zip(bars6, accuracies):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        # 综合性能散点图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 创建散点图，气泡大小表示F1分数
        scatter = ax.scatter(
            results_df['训练时间(s)'],
            results_df['准确率(acc)'],
            s=results_df['F1分数'] * 500,  # 气泡大小
            c=results_df['预测时间/样本(ms)'],
            alpha=0.6,
            cmap='viridis'
        )

        ax.set_xlabel('训练时间(s)')
        ax.set_ylabel('准确率(acc)')
        ax.set_title('模型综合性能对比（气泡大小=F1分数，颜色=预测时间）')

        # 添加模型标签
        for i, row in results_df.iterrows():
            ax.annotate(
                row['模型'],
                (row['训练时间(s)'], row['准确率(acc)']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )

        plt.colorbar(scatter, label='预测时间/样本(ms)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('model_performance_scatter.png', dpi=150, bbox_inches='tight')
        plt.show()

    def analyze_model_parameters(self, results_df):
        """分析模型参数对性能的影响"""
        print("\n5. 模型参数分析")
        print("-" * 60)

        # 分析学习率对性能的影响
        mlp_results = results_df[results_df['学习率'] != '-']
        if len(mlp_results) > 0:
            print("\n学习率对MLP模型性能的影响:")
            for _, row in mlp_results.iterrows():
                print(f"  模型: {row['模型']}, 学习率: {row['学习率']}, "
                      f"准确率: {row['准确率(acc)']:.4f}, 训练时间: {row['训练时间(s)']:.3f}s")

        # 分析隐藏层大小对性能的影响
        hidden_results = results_df[results_df['隐藏层大小'] != '-']
        if len(hidden_results) > 0:
            print("\n隐藏层大小对模型性能的影响:")
            for _, row in hidden_results.iterrows():
                print(f"  模型: {row['模型']}, 隐藏层: {row['隐藏层大小']}, "
                      f"准确率: {row['准确率(acc)']:.4f}, 训练时间: {row['训练时间(s)']:.3f}s")

        # 分析传统模型 vs 神经网络模型
        traditional_models = results_df[results_df['隐藏层大小'] == '-']
        nn_models = results_df[results_df['隐藏层大小'] != '-']

        if len(traditional_models) > 0 and len(nn_models) > 0:
            print("\n传统模型 vs 神经网络模型对比:")
            trad_avg_acc = traditional_models['准确率(acc)'].mean()
            nn_avg_acc = nn_models['准确率(acc)'].mean()
            trad_avg_time = traditional_models['训练时间(s)'].mean()
            nn_avg_time = nn_models['训练时间(s)'].mean()

            print(f"  传统模型平均准确率: {trad_avg_acc:.4f}")
            print(f"  神经网络平均准确率: {nn_avg_acc:.4f}")
            print(f"  传统模型平均训练时间: {trad_avg_time:.3f}s")
            print(f"  神经网络平均训练时间: {nn_avg_time:.3f}s")

            if nn_avg_acc > trad_avg_acc:
                accuracy_gain = (nn_avg_acc - trad_avg_acc) * 100
                print(f"  神经网络准确率提升: +{accuracy_gain:.2f}%")
            else:
                accuracy_loss = (trad_avg_acc - nn_avg_acc) * 100
                print(f"  神经网络准确率降低: -{accuracy_loss:.2f}%")

    def analyze_important_features(self, results_df, vectorizer):
        """分析重要特征"""
        print("\n6. 重要特征分析")
        print("-" * 60)

        # 找出逻辑回归模型
        lr_model = None
        for model_info in self.models:
            if model_info['name'] == '逻辑回归':
                lr_model = model_info['model']
                break

        if lr_model is not None and hasattr(lr_model, 'coef_'):
            # 获取特征重要性
            feature_importance = lr_model.coef_[0]

            # 获取最重要的特征（正面和负面）
            top_n = 10
            top_positive_idx = np.argsort(feature_importance)[-top_n:][::-1]
            top_negative_idx = np.argsort(feature_importance)[:top_n]

            print(f"\n最重要的正面特征（表示好评）:")
            for idx in top_positive_idx:
                if idx < len(self.feature_names):
                    print(f"  {self.feature_names[idx]}: {feature_importance[idx]:.4f}")

            print(f"\n最重要的负面特征（表示差评）:")
            for idx in top_negative_idx:
                if idx < len(self.feature_names):
                    print(f"  {self.feature_names[idx]}: {feature_importance[idx]:.4f}")

    def generate_summary_report(self, results_df):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("实验总结和建议")
        print("=" * 60)

        # 找出最佳模型
        best_acc_idx = results_df['准确率(acc)'].idxmax()
        best_acc = results_df.loc[best_acc_idx]

        best_f1_idx = results_df['F1分数'].idxmax()
        best_f1 = results_df.loc[best_f1_idx]

        fastest_pred_idx = results_df['预测时间/样本(ms)'].idxmin()
        fastest_pred = results_df.loc[fastest_pred_idx]

        fastest_train_idx = results_df['训练时间(s)'].idxmin()
        fastest_train = results_df.loc[fastest_train_idx]

        print(f"\n📊 性能总结:")
        print(
            f"   最高准确率模型: {best_acc['模型']} (准确率: {best_acc['准确率(acc)']:.4f}, 学习率: {best_acc['学习率']}, 隐藏层: {best_acc['隐藏层大小']})")
        print(f"   最高F1分数模型: {best_f1['模型']} (F1分数: {best_f1['F1分数']:.4f})")
        print(f"   最快预测模型: {fastest_pred['模型']} ({fastest_pred['预测时间/样本(ms)']:.2f}ms/样本)")
        print(f"   最快训练模型: {fastest_train['模型']} ({fastest_train['训练时间(s)']:.3f}s)")

        print(f"\n🎯 参数选择建议:")
        if best_acc['学习率'] != '-':
            print(f"   最佳学习率: {best_acc['学习率']}")
        if best_acc['隐藏层大小'] != '-':
            print(f"   最佳隐藏层配置: {best_acc['隐藏层大小']}")

        # 分析学习率建议
        mlp_models = results_df[results_df['学习率'] != '-']
        if len(mlp_models) > 0:
            best_lr_model = mlp_models.loc[mlp_models['准确率(acc)'].idxmax()]
            print(f"   神经网络最佳学习率: {best_lr_model['学习率']} (准确率: {best_lr_model['准确率(acc)']:.4f})")

        print(f"\n💡 模型选择建议:")
        print("   1. 如果追求最高准确率: 选择", best_acc['模型'])
        print("   2. 如果追求平衡性能: 选择", best_f1['模型'])
        print("   3. 如果对实时性要求高: 选择", fastest_pred['模型'])
        print("   4. 如果需要快速迭代: 选择", fastest_train['模型'])
        print("   5. 如果资源有限: 选择朴素贝叶斯或逻辑回归")
        print("   6. 如果数据量大且特征复杂: 考虑神经网络模型")

        # 生成最终推荐
        if best_acc['模型'] == best_f1['模型']:
            print(f"\n🏆 综合推荐模型: {best_acc['模型']} (准确率和F1分数都最佳)")
        else:
            print(f"\n🏆 综合推荐:")
            print(f"   首选: {best_acc['模型']} (准确率最高)")
            print(f"   备选: {best_f1['模型']} (F1分数最高)")
            print(f"   快速选择: {fastest_pred['模型']} (预测最快)")


def main():
    """主函数"""
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建实验运行器
    runner = ExperimentRunner(data_path='文本分类练习.csv')

    # 运行实验
    results_df = runner.run_experiment()

    # 生成总结报告
    runner.generate_summary_report(results_df)

    print("\n" + "=" * 60)
    print("实验完成！")
    print("生成的文件:")
    print("  - model_comparison_results.csv: 模型性能详细结果")
    print("  - model_comparison.png: 模型对比图")
    print("  - model_performance_scatter.png: 综合性能散点图")
    print("=" * 60)


if __name__ == "__main__":
    # # 检查必要的库
    # required_libraries = ['pandas', 'numpy', 'sklearn', 'jieba', 'matplotlib']
    # print("检查必要的库...")
    # for lib in required_libraries:
    #     try:
    #         __import__(lib)
    #         print(f"  ✓ {lib}")
    #     except ImportError:
    #         print(f"  ✗ {lib} 未安装")
    #         if lib == 'sklearn':
    #             print("    请运行: pip install scikit-learn")
    #         elif lib == 'jieba':
    #             print("    请运行: pip install jieba")
    #         else:
    #             print(f"    请运行: pip install {lib}")

    print("\n开始实验...")
    main()
