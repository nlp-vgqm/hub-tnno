import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1. 单个Transformer Encoder层 --------------------------
# BERT的Transformer层 = 多头自注意力 + 前馈网络 + 残差连接 + LayerNorm
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        # 1. 多头自注意力层（PyTorch内置，自动处理QKV投影、多头拆分/拼接）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,  # 输入向量维度（768）
            num_heads=num_heads,    # 注意力头数（12）
            dropout=dropout,        # 注意力权重dropout（防止过拟合）
            batch_first=True        # 输入形状为[batch_size, seq_len, hidden_size]（符合PyTorch习惯）
        )
        
        # 2. 注意力层后的LayerNorm和Dropout
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-12)  # 层归一化（eps防止除0）
        self.dropout1 = nn.Dropout(dropout)
        
        # 3. 前馈网络（FFN）：Linear→GELU→Linear（BERT标准配置）
        self.linear1 = nn.Linear(hidden_size, intermediate_size)  # 768→3072（扩大维度）
        self.linear2 = nn.Linear(intermediate_size, hidden_size)  # 3072→768（还原维度）
        self.gelu = nn.GELU()  # 激活函数（PyTorch 1.10+支持，低版本可用torch.sigmoid近似）
        
        # 4. 前馈网络后的LayerNorm和Dropout
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size] → 示例：[1, 4, 768]
        
        # -------------------------- 第一步：多头自注意力 + 残差连接 --------------------------
        attn_output, _ = self.self_attn(x, x, x)  # 自注意力（Q=K=V=x）
        x = x + self.dropout1(attn_output)        # 残差连接（输入+注意力输出）
        x = self.layer_norm1(x)                   # 层归一化
        
        # -------------------------- 第二步：前馈网络 + 残差连接 --------------------------
        ffn_output = self.linear1(x)              # 768→3072
        ffn_output = self.gelu(ffn_output)        # 非线性激活
        ffn_output = self.dropout2(ffn_output)    # Dropout
        ffn_output = self.linear2(ffn_output)     # 3072→768
        x = x + ffn_output                        # 残差连接
        x = self.layer_norm2(x)                   # 层归一化
        
        return x  # [batch_size, seq_len, hidden_size]


# -------------------------- 2. 完整Transformer Encoder（多层堆叠） --------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        # 堆叠num_layers个Encoder层（BERT-base是6层）
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        for layer in self.layers:
            x = layer(x)  # 逐层传递，更新特征
        return x  # [batch_size, seq_len, hidden_size]（所有token的最终特征）


# -------------------------- 3. BERT风格完整模型（Embedding + Encoder + Pooler） --------------------------
class BertLikeModel(nn.Module):
    def __init__(self, vocab_size=21128,  # bert-base-chinese的词表大小
                 hidden_size=768,
                 num_layers=6,
                 num_heads=12,
                 intermediate_size=3072,
                 max_position_embeddings=512,  # 最大序列长度
                 dropout=0.1):
        super().__init__()
        
        # 1. Embedding层（词嵌入+位置嵌入+类型嵌入，与手动实现一致）
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)  # 词嵌入
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)  # 位置嵌入
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # 句子类型嵌入（0/1）
        
        # Embedding层后归一化和Dropout
        self.embedding_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 2. Transformer Encoder（核心主体）
        self.encoder = TransformerEncoder(num_layers, hidden_size, num_heads, intermediate_size, dropout)
        
        # 3. Pooler层（提取[CLS] token特征，句子级输出）
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 768→768
            nn.Tanh()  # 激活函数，压缩到[-1,1]
        )

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len] → 示例：[1, 4]（1个样本，4个token索引）
        batch_size, seq_len = input_ids.shape
        
        # -------------------------- 第一步：Embedding层 --------------------------
        # 词嵌入：[batch_size, seq_len] → [batch_size, seq_len, hidden_size]
        word_emb = self.word_embeddings(input_ids)
        # 位置嵌入：生成位置索引[0,1,...,seq_len-1]，再取嵌入
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(position_ids)
        # 类型嵌入：单句输入，全0
        token_type_ids = torch.zeros_like(input_ids)  # [batch_size, seq_len]
        type_emb = self.token_type_embeddings(token_type_ids)
        
        # 三种嵌入加和 + 归一化 + Dropout
        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)  # [1,4,768]
        
        # -------------------------- 第二步：Transformer Encoder --------------------------
        sequence_output = self.encoder(embeddings)  # [1,4,768]（所有token的特征）
        
        # -------------------------- 第三步：Pooler层（句子级输出） --------------------------
        cls_token_output = sequence_output[:, 0, :]  # 取第0个token（[CLS]）的特征：[1,768]
        pooler_output = self.pooler(cls_token_output)  # [1,768]
        
        return sequence_output, pooler_output  # 与手动实现输出格式一致

        if __name__ == "__main__":
        # 1. 配置超参数（与之前手动实现一致）
            hidden_size = 768
            num_layers = 6
            num_heads = 12
            vocab_size = 21128  # bert-base-chinese词表大小
            seq_len = 4  # 示例句子长度（4个token）
            
            # 2. 构造输入（与手动实现的x一致）
            x = torch.LongTensor([[2450, 15486, 102, 2110]])  # [batch_size=1, seq_len=4]
            
            # 3. 初始化模型
            model = BertLikeModel(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads
            )
            
            # 4. 前向传播
            model.eval()  # 推理模式（关闭Dropout）
            with torch.no_grad():  # 不计算梯度，加速
                sequence_output, pooler_output = model(x)
            
            # 5. 打印输出形状（与手动实现对比）
            print("sequence_output形状：", sequence_output.shape)  # 预期：torch.Size([1, 4, 768])
            print("pooler_output形状：", pooler_output.shape)      # 预期：torch.Size([1, 768])
            
            # 打印部分输出（验证模型正常运行）
            print("\nsequence_output前2个token的前5维特征：")
            print(sequence_output[0, :2, :5])
            print("\npooler_output前5维特征：")
            print(pooler_output[0, :5])
