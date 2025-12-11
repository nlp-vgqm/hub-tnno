import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询向量变换
        self.W_k = nn.Linear(d_model, d_model)  # 键向量变换
        self.W_v = nn.Linear(d_model, d_model)  # 值向量变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用掩码（如需要）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权和
        output = torch.matmul(attn_weights, V)
        return output

    def split_heads(self, x):
        # 将输入张量分割为多个头
        # 输入形状: (batch_size, seq_len, d_model)
        # 输出形状: (batch_size, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 合并多个头
        # 输入形状: (batch_size, num_heads, seq_len, d_k)
        # 输出形状: (batch_size, seq_len, d_model)
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 线性变换并分割头
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并头并线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFFN(nn.Module):
    """位置全连接前馈网络"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一个全连接层
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二个全连接层
        self.activation = nn.GELU()  # GELU激活函数

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 子层
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFFN(d_model, d_ff)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)  # 残差连接
        x = self.norm1(x)  # 层归一化

        # 前馈网络子层
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(ffn_output)  # 残差连接
        x = self.norm2(x)  # 层归一化

        return x


# 示例用法
if __name__ == "__main__":
    # 设置超参数
    d_model = 512  # 模型维度
    num_heads = 8  # 注意力头数
    d_ff = 2048  # 前馈网络隐藏层维度
    batch_size = 2  # 批次大小
    seq_len = 10  # 序列长度

    # 创建示例输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 创建Transformer层
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff)

    # 前向传播
    output = transformer_layer(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("输出示例:")
    print(output[0, 0, :5])  # 打印第一个样本的第一个位置的5个特征