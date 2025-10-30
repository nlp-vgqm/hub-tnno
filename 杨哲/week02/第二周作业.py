import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class MaxValueClassifier(nn.Module):
    def __init__(self, input_dim=5, output_dim=5):
        super(MaxValueClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def generate_data(num_samples=10000):
    """生成训练数据：五维随机向量，标签是最大值所在的维度"""
    # 生成随机数据
    X = np.random.randn(num_samples, 5)

    # 找到每个样本中最大值的位置
    y = np.argmax(X, axis=1)

    return torch.FloatTensor(X), torch.LongTensor(y)


def train_model():
    # 生成数据
    X_train, y_train = generate_data(8000)
    X_test, y_test = generate_data(2000)

    # 创建模型
    model = MaxValueClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    epochs = 100
    batch_size = 64
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("开始训练...")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        # 批量训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()

        # 计算训练准确率
        train_accuracy = 100 * epoch_correct / epoch_total
        train_losses.append(epoch_loss / (len(X_train) // batch_size))
        train_accuracies.append(train_accuracy)

        # 测试集准确率
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = 100 * (test_predicted == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / (len(X_train) // batch_size):.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

    return model, train_losses, train_accuracies, test_accuracies


def evaluate_model(model, num_samples=100):
    """评估模型性能"""
    model.eval()
    X_test, y_test = generate_data(num_samples)

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / num_samples

        print(f"\n模型评估结果:")
        print(f"测试样本数: {num_samples}")
        print(f"准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # 显示一些预测示例
        print("\n预测示例:")
        print("输入向量 -> 真实类别 | 预测类别")
        print("-" * 40)
        for i in range(min(10, num_samples)):
            input_vec = X_test[i].numpy()
            true_class = y_test[i].item()
            pred_class = predicted[i].item()
            print(f"{input_vec} -> {true_class} | {pred_class} {'✓' if true_class == pred_class else '✗'}")


def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 运行训练和评估
if __name__ == "__main__":
    # 训练模型
    model, train_losses, train_accuracies, test_accuracies = train_model()

    # 评估模型
    evaluate_model(model)

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accuracies, test_accuracies)

    # 测试新样本
    print("\n新样本测试:")
    model.eval()
    with torch.no_grad():
        # 生成新的测试样本
        new_samples = torch.FloatTensor(np.random.randn(5, 5))
        outputs = model(new_samples)
        _, predictions = torch.max(outputs, 1)

        for i, sample in enumerate(new_samples):
            max_idx = torch.argmax(sample).item()
            pred_idx = predictions[i].item()
            print(
                f"样本 {i + 1}: {sample.numpy()} -> 最大值在维度 {max_idx}, 预测类别: {pred_idx} {'✓' if max_idx == pred_idx else '✗'}")
