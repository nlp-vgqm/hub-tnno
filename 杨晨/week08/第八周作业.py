import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity


class TripletDataset(Dataset):
    def __init__(self, data_path):
        """
        三元组数据集加载器
        数据格式: (anchor, positive, negative)
        """
        self.triplets = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                anchor, pos, neg = line.strip().split('\t')
                self.triplets.append((anchor, pos, neg))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


class SentenceEncoder(nn.Module):
    """
    句子编码器（参考文档6第13页）
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # 输入x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        output, _ = self.rnn(embedded)  # [batch_size, seq_len, hidden_size]
        # 取最后一个时间步的输出
        last_output = output[:, -1, :]  # [batch_size, hidden_size]
        return self.fc(last_output)  # [batch_size, hidden_size]


class TripletLoss(nn.Module):
    """
    三元组损失函数（参考文档8第38页）
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        计算三元组损失
        anchor: 锚点句子嵌入 [batch_size, hidden_size]
        positive: 正样本句子嵌入 [batch_size, hidden_size]
        negative: 负样本句子嵌入 [batch_size, hidden_size]
        """
        # 计算余弦距离（参考文档8第38页）
        pos_dist = 1 - torch.cosine_similarity(anchor, positive)
        neg_dist = 1 - torch.cosine_similarity(anchor, negative)

        # 三元组损失公式（参考文档8第38页）
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


def sentence_to_tensor(sentence, vocab, max_len=50):
    """
    将句子转换为张量（参考文档6第11页）
    """
    tokens = sentence.split()[:max_len]
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    # 填充到固定长度
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


def main():
    # 参数设置（参考文档6第18页）
    vocab_size = 10000  # 词汇表大小
    embedding_dim = 128  # 词向量维度
    hidden_size = 256  # RNN隐藏层大小
    batch_size = 32
    num_epochs = 10
    margin = 0.5

    # 创建词汇表（实际应用中应从数据构建）
    vocab = {'<pad>': 0, '<unk>': 1}
    # 这里简化处理，实际应基于训练数据构建词汇表

    # 加载数据集
    train_dataset = TripletDataset('train_triplets.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型和损失
    model = SentenceEncoder(vocab_size, embedding_dim, hidden_size)
    criterion = TripletLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # 将文本转换为张量
            anchor_tensors = []
            pos_tensors = []
            neg_tensors = []

            for anchor, pos, neg in batch:
                anchor_tensors.append(sentence_to_tensor(anchor, vocab))
                pos_tensors.append(sentence_to_tensor(pos, vocab))
                neg_tensors.append(sentence_to_tensor(neg, vocab))

            anchor_batch = torch.stack(anchor_tensors)
            pos_batch = torch.stack(pos_tensors)
            neg_batch = torch.stack(neg_tensors)

            # 前向传播
            anchor_emb = model(anchor_batch)
            pos_emb = model(pos_batch)
            neg_emb = model(neg_batch)

            # 计算损失
            loss = criterion(anchor_emb, pos_emb, neg_emb)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'triplet_model.pth')
    print('训练完成，模型已保存！')

    # 测试相似度计算
    model.eval()
    test_sentences = [
        "如何还款",
        "怎么还款",
        "如何开户"
    ]

    embeddings = []
    for sent in test_sentences:
        tensor = sentence_to_tensor(sent, vocab).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor)
        embeddings.append(emb.squeeze().numpy())

    # 计算相似度（参考文档8第38页）
    sim_matrix = cosine_similarity(embeddings)
    print("\n句子相似度矩阵:")
    print("句子1: 如何还款")
    print("句子2: 怎么还款")
    print("句子3: 如何开户")
    print(f"相似度1-2: {sim_matrix[0][1]:.4f}")
    print(f"相似度1-3: {sim_matrix[0][2]:.4f}")


if __name__ == "__main__":
    main()
# 以上代码由AI生成，由于本人完全不懂Python，故无法理解上述内容
