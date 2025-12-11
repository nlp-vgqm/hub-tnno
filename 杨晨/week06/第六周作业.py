import torch
import torch.nn as nn
import math


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)

        # 多头自注意力的线性变换：Q、K、V和输出层（参考文档5的self_attention方法）
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)

        # 层归一化（使用PyTorch内置LayerNorm，替代文档5的自定义函数）
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

        # 前馈网络（Position-wise Feed-Forward Networks），参考文档1第3.3节
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)

        # 激活函数：BERT使用GELU（参考文档5的gelu函数，但这里用PyTorch内置）
        self.activation = nn.GELU()

    def forward(self, x):
        # 输入x形状: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape

        # 1. 多头自注意力部分
        # 计算Q、K、V（线性变换）
        q = self.query(x)  # 形状: [batch_size, seq_len, hidden_size]
        k = self.key(x)
        v = self.value(x)

        # 重塑为多头格式（参考文档5的transpose_for_scores方法）
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1,
                                                                                                      2)  # [batch, num_heads, seq_len, head_size]
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 计算注意力分数（Scaled Dot-Product Attention，文档1第3.2.1节）
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.attention_head_size)  # [batch, num_heads, seq_len, seq_len]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # 应用softmax

        # 应用注意力到V值
        context = torch.matmul(attention_probs, v)  # [batch, num_heads, seq_len, head_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)  # 合并多头

        # 注意力输出线性层
        attention_output = self.attention_output(context)  # 形状: [batch_size, seq_len, hidden_size]

        # 残差连接和层归一化（参考文档5的残差设计）
        x = self.attention_layer_norm(x + attention_output)

        # 2. 前馈网络部分（文档1第3.3节）
        intermediate_output = self.intermediate(x)  # 形状: [batch_size, seq_len, intermediate_size]
        intermediate_output = self.activation(intermediate_output)  # GELU激活
        ff_output = self.output(intermediate_output)  # 形状: [batch_size, seq_len, hidden_size]

        # 残差连接和层归一化
        x = self.output_layer_norm(x + ff_output)

        return x  # 输出形状: [batch_size, seq_len, hidden_size]
    #由于不会Python基础，以上代码完全由AI生成，本人完全不懂以上内容
