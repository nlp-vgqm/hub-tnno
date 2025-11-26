#coding:utf8

import torch
import torch.nn as nn
import numpy as np

'''
基于pytorch实现单层transformer结构
'''

class TransformerLayer(nn.Module):
    def __init__(self, vocab_size, segment_size=2, position_size=512):
        super(TransformerLayer, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.segment_embedding = nn.Embedding(segment_size, self.hidden_size)
        self.position_embedding = nn.Embedding(position_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)

        # Attention
        self.layer_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = torch.softmax
        self.attention_output = nn.Linear(self.hidden_size, self.hidden_size)

        # Feed-forward network
        self.layer_ff1 = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.gelu = nn.GELU()
        self.layer_ff2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, segment_ids=None):
        batch_size, seq_length = input_ids.size()

        # 初始化 segment_ids 为全0，如果未提供
        if segment_ids is None:
            segment_ids = torch.zeros(input_ids.shape, dtype=torch.long)

        # Step 1: Embedding
        embedding = (
                self.token_embedding(input_ids) +  # (B, L, H)
                self.segment_embedding(segment_ids) +  # (B, L, H)
                self.position_embedding(torch.arange(seq_length))  # (L, H)
        )
        embedding = self.layer_norm(embedding)
        embedding = self.dropout(embedding)

        # Step 2: Self-Attention
        q = self.layer_q(embedding)  # (B, L ,H)
        k = self.layer_k(embedding)
        v = self.layer_v(embedding)

        # multi-head: (B, L ,H)  ->  (B, L, head_num, H/head_num)  ->  (B, head_num, L, H/head_num)
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        #
        qk = torch.matmul(q, k.transpose(-1, -2))
        qk = qk / (self.head_dim ** 0.5)
        attention_probs = self.softmax(qk, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        atten_output = self.attention_output(context)
        atten_output = self.dropout(atten_output)

        # residual connection & layer norm
        atten_residual = self.layer_norm(embedding + atten_output)

        # Step 3: Feed-Forward Network
        ff_output = self.layer_ff1(atten_residual)
        ff_output = self.gelu(ff_output)
        ff_output = self.layer_ff2(ff_output)
        ff_output = self.dropout(ff_output)

        # residual connection & layer norm
        output = self.layer_norm2(ff_output + atten_residual)
        return output



# 测试模型
def test_transformer():

    # 示例调用
    model = TransformerLayer(vocab_size=30522)  # BERT-base 词汇表大小
    # 生成一个形状为 (2, 10) 的张量，其中每个元素是在 [0, 30521] 范围内的随机整数。
    input_ids = torch.randint(0, 30522, (2, 10))  # 批大小=2，序列长=10

    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

    output = model(input_ids, segment_ids)
    print(output.shape)  # 应输出 torch.Size([2, 10, 768])


# 运行测试
if __name__ == "__main__":
    test_transformer()




















