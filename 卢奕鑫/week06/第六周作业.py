import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask（如果有）
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到value上
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出线性变换
        output = self.w_o(attn_output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 也可以用ReLU

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络子层 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力子层（带mask）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 交叉注意力子层
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


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
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # 词嵌入和位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_embedded = self.dropout(self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        encoder_output = src_embedded

        for layer in self.encoder_layers:
            encoder_output, _ = layer(encoder_output, src_mask)

        # 解码器
        tgt_embedded = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        decoder_output = tgt_embedded

        for layer in self.decoder_layers:
            decoder_output, _, _ = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出
        output = self.output_layer(decoder_output)

        return output


# 使用示例
if __name__ == "__main__":
    # 超参数
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    batch_size = 32
    seq_len = 50

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )

    # 示例输入
    src = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    # 前向传播
    output = model(src, tgt)

    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
