# -*- coding: utf-8 -*-

"""
基于BERT的自回归语言模型（Encoder-Decoder架构）
使用BERT作为Encoder，添加Decoder用于自回归生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel, BertTokenizer


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)


class DecoderLayer(nn.Module):
    """Decoder层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class BertEncoderDecoder(nn.Module):
    """基于BERT的Encoder-Decoder模型，用于自回归文本生成"""
    
    def __init__(self, config):
        super(BertEncoderDecoder, self).__init__()
        self.config = config
        
        # BERT作为Encoder
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_hidden_size = self.bert.config.hidden_size
        
        # Decoder部分
        self.decoder_embedding = nn.Embedding(
            self.bert.config.vocab_size,
            bert_hidden_size
        )
        self.pos_encoding = PositionalEncoding(bert_hidden_size, config["output_max_length"], config["dropout"])
        
        # Decoder层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                bert_hidden_size,
                config["num_heads"],
                bert_hidden_size * 4,  # FFN维度
                config["dropout"]
            )
            for _ in range(config["num_decoder_layers"])
        ])
        
        # 输出层
        self.output_projection = nn.Linear(bert_hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(config["dropout"])
        
        # Tokenizer（用于生成时使用）
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码（causal mask），用于自回归生成"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, encoder_input_ids, encoder_attention_mask, 
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        """
        前向传播
        Args:
            encoder_input_ids: Encoder输入 (batch_size, src_len)
            encoder_attention_mask: Encoder注意力掩码 (batch_size, src_len)
            decoder_input_ids: Decoder输入 (batch_size, tgt_len)，训练时提供
            decoder_attention_mask: Decoder注意力掩码 (batch_size, tgt_len)
            labels: 标签 (batch_size, tgt_len)，用于计算损失
        Returns:
            如果提供了labels，返回损失值
            否则返回logits (batch_size, tgt_len, vocab_size)
        """
        batch_size = encoder_input_ids.size(0)
        
        # Encoder: 使用BERT编码输入
        encoder_outputs = self.bert(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask
        )
        encoder_output = encoder_outputs.last_hidden_state  # (batch_size, src_len, hidden_size)
        
        # 准备Encoder的mask（用于cross-attention）
        if encoder_attention_mask is not None:
            src_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
        else:
            src_mask = None
        
        if decoder_input_ids is not None:
            # 训练模式：使用teacher forcing
            tgt_len = decoder_input_ids.size(1)
            
            # Decoder embedding
            decoder_emb = self.decoder_embedding(decoder_input_ids)  # (batch_size, tgt_len, hidden_size)
            decoder_emb = decoder_emb.transpose(0, 1)  # (tgt_len, batch_size, hidden_size)
            decoder_emb = self.pos_encoding(decoder_emb)
            decoder_emb = decoder_emb.transpose(0, 1)  # (batch_size, tgt_len, hidden_size)
            decoder_emb = self.dropout(decoder_emb)
            
            # 生成因果掩码（防止看到未来信息）
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(decoder_input_ids.device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
            
            # 如果提供了decoder_attention_mask，需要结合使用
            if decoder_attention_mask is not None:
                # 扩展decoder_attention_mask以匹配tgt_mask的形状
                decoder_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
                # 结合因果掩码和padding掩码
                tgt_mask = tgt_mask.masked_fill(decoder_mask == 0, float('-inf'))
            
            # 通过Decoder层
            decoder_output = decoder_emb
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
            # 输出投影
            logits = self.output_projection(decoder_output)  # (batch_size, tgt_len, vocab_size)
            
            if labels is not None:
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return loss
            else:
                return logits
        else:
            # 推理模式：自回归生成
            return self.generate(encoder_input_ids, encoder_attention_mask)
    
    def generate(self, encoder_input_ids, encoder_attention_mask, max_length=None, beam_size=1):
        """
        自回归生成
        Args:
            encoder_input_ids: Encoder输入
            encoder_attention_mask: Encoder注意力掩码
            max_length: 最大生成长度
            beam_size: Beam search大小
        Returns:
            生成的token ids
        """
        if max_length is None:
            max_length = self.config["output_max_length"]
        
        batch_size = encoder_input_ids.size(0)
        device = encoder_input_ids.device
        
        # Encoder编码
        encoder_outputs = self.bert(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask
        )
        encoder_output = encoder_outputs.last_hidden_state
        
        # 准备Encoder mask
        if encoder_attention_mask is not None:
            src_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            src_mask = None
        
        # 初始化：使用[CLS] token作为起始
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.cls_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 自回归生成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_length - 1):
            # 如果所有序列都已完成，提前退出
            if finished.all():
                break
            
            # Decoder embedding
            decoder_emb = self.decoder_embedding(decoder_input_ids)
            tgt_len = decoder_emb.size(1)
            decoder_emb = decoder_emb.transpose(0, 1)
            decoder_emb = self.pos_encoding(decoder_emb)
            decoder_emb = decoder_emb.transpose(0, 1)
            
            # 生成因果掩码
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
            
            # 通过Decoder层
            decoder_output = decoder_emb
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
            # 获取下一个token的logits
            logits = self.output_projection(decoder_output)  # (batch_size, tgt_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 选择下一个token（贪心解码）
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
            
            # 检查是否遇到[SEP] token（结束）
            finished = finished | (next_token_id.squeeze(1) == self.tokenizer.sep_token_id)
            
            # 添加到序列
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)
        
        return decoder_input_ids


def choose_optimizer(config, model):
    """选择优化器"""
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支持的优化器: {optimizer}")


if __name__ == "__main__":
    from config import Config
    model = BertEncoderDecoder(Config)
    print("模型创建成功！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
