import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output, attn_probs

    def split_heads(self, x):
        """将张量分割为多头"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """合并多头"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割多头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算注意力
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        output = self.combine_heads(attn_output)

        # 最终线性变换
        output = self.W_o(output)

        return output, attn_probs


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """单个Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层（残差连接+层归一化）
        attn_output, attn_probs = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络子层（残差连接+层归一化）
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_probs


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class SimpleTransformer(nn.Module):
    """完整的单层Transformer模型"""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, max_len=512, dropout=0.1):
        super(SimpleTransformer, self).__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer编码器层
        self.encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq):
        """创建填充掩码"""
        # seq: (batch_size, seq_len)
        mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return mask

    def forward(self, src, src_mask=None):
        """
        前向传播
        Args:
            src: 输入序列 (batch_size, seq_len)
            src_mask: 注意力掩码
        """
        # 词嵌入
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)

        # 位置编码
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # 如果没有提供掩码，创建填充掩码
        if src_mask is None:
            src_mask = self.create_padding_mask(src)

        # 通过Transformer编码器层
        encoder_output, attn_probs = self.encoder_layer(x, src_mask)

        # 输出层
        output = self.fc_out(encoder_output)

        return output, attn_probs


# 使用示例
def main():
    # 超参数设置
    batch_size = 4
    seq_len = 10
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 创建模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )

    # 创建模拟输入
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("输入形状:", src.shape)
    print("模型架构:")
    print(model)

    # 前向传播
    output, attn_probs = model(src)

    print("\n输出形状:", output.shape)  # (batch_size, seq_len, vocab_size)
    print("注意力权重形状:", attn_probs.shape)  # (batch_size, num_heads, seq_len, seq_len)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试掩码功能
    print("\n测试掩码功能...")
    # 创建一个示例掩码（第一个序列的前3个位置是有效的）
    test_mask = torch.ones(batch_size, 1, 1, seq_len)
    test_mask[0, :, :, 3:] = 0  # 屏蔽第一个样本的后7个位置

    output_with_mask, attn_with_mask = model(src, test_mask)
    print("使用掩码的输出形状:", output_with_mask.shape)


if __name__ == "__main__":
    main()