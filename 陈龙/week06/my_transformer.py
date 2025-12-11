import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SelfAttention(nn.Module):
    """手动实现的自注意力机制"""

    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性变换层：Q, K, V 投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)

        self.scale = math.sqrt(self.d_k)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape

        # 线性投影得到Q, K, V
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_k]

        # 计算注意力分数 Q*K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, n_heads, seq_len, seq_len]

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch_size, n_heads, seq_len, d_k]

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        # 输出投影
        output = self.w_o(attn_output)

        return output


class OneLayerTransformer(nn.Module):
    def __init__(self, input_size, n_heads=8):
        super(OneLayerTransformer, self).__init__()
        self.self_attention = SelfAttention(input_size, n_heads)
        self.layer_norm = LayerNorm(input_size)
        self.feed_forward = FeedForward(input_size, input_size * 2)
        self.ff_layer_norm = LayerNorm(input_size)

    def forward(self, x):
        x = self.layer_norm(x + self.self_attention(x))
        return self.ff_layer_norm(x + self.feed_forward(x))


# 测试代码
def test_transformer():
    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # 创建模型
    transformer_layer = OneLayerTransformer(d_model, n_heads)

    # 创建输入数据
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = transformer_layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("模型参数数量:", sum(p.numel() for p in transformer_layer.parameters()))


if __name__ == "__main__":
    test_transformer()
