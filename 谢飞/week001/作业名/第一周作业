import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=5):
        super(MultiClassClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def generate_data(num_samples=10000):
    """生成训练数据：五维随机向量，标签是最大值所在的维度"""
    # 生成随机数据
    X = np.random.randn(num_samples, 5)
    
    # 找到每个样本中最大值所在的维度（0-4）
    y = np.argmax(X, axis=1)
    
    return torch.FloatTensor(X), torch.LongTensor(y)

def train_model():
    # 生成数据
    X, y = generate_data(10000)
    
    # 划分训练集和测试集
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 创建模型
    model = MultiClassClassifier(input_dim=5, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    epochs = 100
    batch_size = 32
    train_losses = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # 批量训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # 计算平均训练损失
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # 评估测试集准确率
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
            test_accuracies.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    return model, train_losses, test_accuracies

def plot_training_curves(train_losses, test_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 训练损失
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 测试准确率
    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_model(model, num_test_samples=100):
    """测试模型性能"""
    model.eval()
    
    # 生成测试数据
    X_test, y_test = generate_data(num_test_samples)
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f"测试准确率: {accuracy:.4f}")
        
        # 显示一些预测示例
        print("\n预测示例:")
        print("真实标签 -> 预测标签")
        for i in range(min(10, num_test_samples)):
            print(f"    {y_test[i].item()}    ->    {predicted[i].item()}")

def predict_single_sample(model, sample=None):
    """对单个样本进行预测"""
    if sample is None:
        # 生成随机样本
        sample = np.random.randn(5)
    
    model.eval()
    with torch.no_grad():
        sample_tensor = torch.FloatTensor(sample).unsqueeze(0)
        output = model(sample_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        
        print(f"输入向量: {sample}")
        print(f"最大值位置: {np.argmax(sample)}")
        print(f"预测类别: {predicted_class}")
        print(f"各类别概率: {probabilities.squeeze().numpy()}")
        
        return predicted_class

if __name__ == "__main__":
    # 训练模型
    print("开始训练模型...")
    model, train_losses, test_accuracies = train_model()
    
    # 绘制训练曲线
    plot_training_curves(train_losses, test_accuracies)
    
    # 测试模型
    print("\n模型测试结果:")
    test_model(model)
    
    # 单样本预测示例
    print("\n单样本预测示例:")
    predict_single_sample(model)
    
    # 手动测试一些边界情况
    print("\n边界情况测试:")
    
    # 情况1: 明显最大值在第一个位置
    sample1 = [10.0, 1.0, 2.0, 3.0, 4.0]
    predict_single_sample(model, sample1)
    
    # 情况2: 最大值在最后一个位置
    sample2 = [1.0, 2.0, 3.0, 4.0, 10.0]
    predict_single_sample(model, sample2)
    
    # 情况3: 接近的值
    sample3 = [5.0, 5.1, 4.9, 4.8, 4.7]
    predict_single_sample(model, sample3)
