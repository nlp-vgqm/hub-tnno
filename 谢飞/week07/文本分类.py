"""
电商评论分类：好评/差评
实现多种模型对比，包括准确率和预测速度
"""

import pandas as pd
import numpy as np
import time
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost，如果没有则使用GradientBoosting
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

class TextClassifier:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.results = []
        
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', str(text))
        return text.strip()
    
    def load_data(self, filepath):
        """加载数据"""
        print("=" * 60)
        print("1. 数据加载")
        print("=" * 60)
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"数据总量: {len(df)} 条")
        return df
    
    def analyze_data(self, df):
        """数据分析"""
        print("\n" + "=" * 60)
        print("2. 数据分析")
        print("=" * 60)
        
        # 文本预处理
        df['review_clean'] = df['review'].apply(self.preprocess_text)
        
        # 正负样本统计
        label_counts = df['label'].value_counts()
        print(f"\n标签分布:")
        print(f"  好评 (label=1): {label_counts.get(1, 0)} 条 ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  差评 (label=0): {label_counts.get(0, 0)} 条 ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
        
        # 文本长度统计
        df['text_length'] = df['review_clean'].apply(len)
        print(f"\n文本长度统计:")
        print(f"  平均长度: {df['text_length'].mean():.2f} 字符")
        print(f"  最短长度: {df['text_length'].min()} 字符")
        print(f"  最长长度: {df['text_length'].max()} 字符")
        print(f"  中位数长度: {df['text_length'].median():.2f} 字符")
        
        # 显示样本示例
        print(f"\n样本示例:")
        print("好评示例:")
        for i, row in df[df['label'] == 1].head(2).iterrows():
            print(f"  {row['review'][:50]}...")
        print("差评示例:")
        for i, row in df[df['label'] == 0].head(2).iterrows():
            print(f"  {row['review'][:50]}...")
        
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """划分训练集和验证集"""
        print("\n" + "=" * 60)
        print("3. 训练集/验证集划分")
        print("=" * 60)
        
        X = df['review_clean'].values
        y = df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"训练集: {len(X_train)} 条 ({len(X_train)/len(df)*100:.2f}%)")
        print(f"验证集: {len(X_test)} 条 ({len(X_test)/len(df)*100:.2f}%)")
        print(f"训练集标签分布: 好评={np.sum(y_train==1)}, 差评={np.sum(y_train==0)}")
        print(f"验证集标签分布: 好评={np.sum(y_test==1)}, 差评={np.sum(y_test==0)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """训练多种模型"""
        print("\n" + "=" * 60)
        print("4. 模型训练与评估")
        print("=" * 60)
        
        # 特征提取
        print("\n特征提取中...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # 增加特征数
            ngram_range=(1, 2),
            min_df=1,  # 降低最小文档频率，保留更多特征
            max_df=0.95,
            sublinear_tf=True  # 使用对数缩放
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        print(f"特征维度: {X_train_vec.shape[1]}")
        
        # 定义模型列表
        models_config = [
            {
                'name': '朴素贝叶斯',
                'model': MultinomialNB(alpha=0.1, fit_prior=False),  # 调整alpha，不使用先验
                'params': {
                    'learning_rate': 'N/A',
                    'hidden_layer_sizes': 'N/A',
                    'n_estimators': 'N/A',
                    'max_depth': 'N/A',
                    'C': 'N/A',
                    'kernel': 'N/A',
                    'alpha': 0.1
                }
            },
            {
                'name': '逻辑回归',
                'model': LogisticRegression(max_iter=2000, random_state=42, C=10.0, class_weight='balanced', solver='liblinear'),
                'params': {
                    'learning_rate': 'N/A',
                    'hidden_layer_sizes': 'N/A',
                    'n_estimators': 'N/A',
                    'max_depth': 'N/A',
                    'C': 10.0,
                    'kernel': 'N/A'
                }
            },
            {
                'name': 'SVM (线性)',
                'model': SVC(kernel='linear', random_state=42, C=10.0, class_weight='balanced', probability=True),
                'params': {
                    'learning_rate': 'N/A',
                    'hidden_layer_sizes': 'N/A',
                    'n_estimators': 'N/A',
                    'max_depth': 'N/A',
                    'C': 10.0,
                    'kernel': 'linear'
                }
            },
            {
                'name': '神经网络 (MLP)',
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50), 
                    max_iter=500, 
                    random_state=42, 
                    learning_rate='adaptive',
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                'params': {
                    'learning_rate': 'adaptive',
                    'hidden_layer_sizes': (100, 50),
                    'n_estimators': 'N/A',
                    'max_depth': 'N/A',
                    'C': 'N/A',
                    'kernel': 'N/A'
                }
            }
        ]
        
        # 添加支持更多参数的模型
        if HAS_XGBOOST:
            models_config.append({
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'learning_rate': 0.1,
                    'hidden_layer_sizes': 'N/A',
                    'n_estimators': 100,
                    'max_depth': 5,
                    'C': 'N/A',
                    'kernel': 'N/A'
                }
            })
        else:
            # 使用GradientBoosting作为替代
            models_config.append({
                'name': '梯度提升',
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    subsample=0.8  # 添加子采样
                ),
                'params': {
                    'learning_rate': 0.1,
                    'hidden_layer_sizes': 'N/A',
                    'n_estimators': 100,
                    'max_depth': 5,
                    'C': 'N/A',
                    'kernel': 'N/A'
                }
            })
        
        # 添加随机森林模型
        models_config.append({
            'name': '随机森林',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # 添加类别权重
            ),
            'params': {
                'learning_rate': 'N/A',
                'hidden_layer_sizes': 'N/A',
                'n_estimators': 100,
                'max_depth': 20,
                'C': 'N/A',
                'kernel': 'N/A'
            }
        })
        
        # 计算样本权重（用于不支持class_weight的模型）
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # 训练和评估每个模型
        for config in models_config:
            print(f"\n训练模型: {config['name']}")
            model = config['model']
            
            # 训练时间
            start_time = time.time()
            # 对于朴素贝叶斯，使用样本权重
            if config['name'] == '朴素贝叶斯':
                # 朴素贝叶斯不支持sample_weight，使用fit_prior=False和调整alpha
                model.fit(X_train_vec, y_train)
            elif hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                # 如果模型支持sample_weight，使用它
                try:
                    model.fit(X_train_vec, y_train, sample_weight=sample_weights)
                except:
                    model.fit(X_train_vec, y_train)
            else:
                model.fit(X_train_vec, y_train)
            train_time = time.time() - start_time
            
            # 预测准确率
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # 诊断信息：显示预测分布和详细指标
            from collections import Counter
            pred_dist = Counter(y_pred)
            true_dist = Counter(y_test)
            print(f"  真实标签分布: {dict(true_dist)}")
            print(f"  预测标签分布: {dict(pred_dist)}")
            print(f"  F1分数: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
            
            # 预测速度测试（预测100条数据，多次测量取平均）
            test_samples = X_test_vec[:100]
            # 预热
            _ = model.predict(test_samples)
            # 多次测量取平均，提高精度
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                _ = model.predict(test_samples)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            predict_time = np.mean(times) / 1000  # 转回秒用于显示
            
            # 保存模型和结果
            self.models[config['name']] = model
            params = config['params']
            self.results.append({
                'Model': config['name'],
                'Learning_Rate': str(params.get('learning_rate', 'N/A')),
                'Hidden_Size': str(params.get('hidden_layer_sizes', 'N/A')),
                'N_Estimators': str(params.get('n_estimators', 'N/A')),
                'Max_Depth': str(params.get('max_depth', 'N/A')),
                'C': str(params.get('C', 'N/A')),
                'Kernel': str(params.get('kernel', 'N/A')),
                'acc': f"{accuracy:.4f}",
                'f1': f"{f1:.4f}",
                'precision': f"{precision:.4f}",
                'recall': f"{recall:.4f}",
                'time(预测100条耗时)': f"{np.mean(times):.2f}ms",
                'train_time': train_time
            })
            
            print(f"  准确率: {accuracy:.4f}, F1: {f1:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")
            print(f"  训练时间: {train_time:.2f}秒")
            print(f"  预测100条耗时: {np.mean(times):.2f}ms (平均10次)")
        
        return self.results
    
    def print_results_table(self):
        """打印结果表格"""
        print("\n" + "=" * 60)
        print("5. 模型对比结果表格")
        print("=" * 60)
        
        df_results = pd.DataFrame(self.results)
        
        # 选择要显示的列，按照要求格式
        display_cols = ['Model', 'Learning_Rate', 'Hidden_Size', 'N_Estimators', 'Max_Depth', 'C', 'Kernel', 
                       'acc', 'f1', 'precision', 'recall', 'time(预测100条耗时)']
        df_display = df_results[display_cols].copy()
        
        # 重命名列名，使其更清晰
        df_display.columns = ['模型', '学习率', '隐藏层大小', '树数量', '最大深度', 'C参数', '核函数', 
                             '准确率', 'F1分数', '精确率', '召回率', '预测100条耗时(ms)']
        
        print("\n" + "=" * 120)
        print(df_display.to_string(index=False))
        print("=" * 120)
        
        # 保存到CSV（使用原始列名）
        output_file = '模型对比结果.csv'
        df_results[display_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        
        # 找出最佳模型（按F1分数，因为数据不平衡）
        df_results['f1_float'] = df_results['f1'].astype(float)
        df_results['acc_float'] = df_results['acc'].astype(float)
        
        best_f1_model = df_results.loc[df_results['f1_float'].idxmax()]
        best_acc_model = df_results.loc[df_results['acc_float'].idxmax()]
        
        print(f"\n最佳模型（按F1分数）: {best_f1_model['Model']} (F1: {best_f1_model['f1']}, 准确率: {best_f1_model['acc']})")
        print(f"最佳模型（按准确率）: {best_acc_model['Model']} (准确率: {best_acc_model['acc']}, F1: {best_acc_model['f1']})")
        
        # 统计信息
        print(f"\n模型统计:")
        print(f"  平均准确率: {df_results['acc_float'].mean():.4f}")
        print(f"  平均F1分数: {df_results['f1_float'].mean():.4f}")
        # 计算最快预测速度
        time_values = df_results['time(预测100条耗时)'].str.replace('ms', '').astype(float)
        fastest_model = df_results.loc[time_values.idxmin(), 'Model']
        fastest_time = time_values.min()
        print(f"  最快预测速度: {fastest_model} ({fastest_time:.2f}ms)")
        
        return df_display

def main():
    """主函数"""
    classifier = TextClassifier()
    
    # 1. 加载数据
    df = classifier.load_data('文本分类练习.csv')
    
    # 2. 数据分析
    df = classifier.analyze_data(df)
    
    # 3. 划分训练集和验证集
    X_train, X_test, y_train, y_test = classifier.split_data(df)
    
    # 4. 训练和评估模型
    results = classifier.train_models(X_train, X_test, y_train, y_test)
    
    # 5. 输出结果表格
    results_table = classifier.print_results_table()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()

