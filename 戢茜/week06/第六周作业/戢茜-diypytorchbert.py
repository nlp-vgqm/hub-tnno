class BertConfig:
    """BERT模型配置类"""

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

    @classmethod
    def from_pretrained(cls, model_name="bert-base-uncased"):
        """从预训练模型名称获取配置"""
        if model_name == "bert-base-uncased":
            return cls(
                vocab_size=30522,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512
            )
        elif model_name == "bert-large-uncased":
            return cls(
                vocab_size=30522,
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                max_position_embeddings=512
            )
        else:
            return cls()


import torch
import torch.nn as nn
import math


class BertEmbeddings(nn.Module):
    """BERT嵌入层：词嵌入 + 位置嵌入 + 段落类型嵌入"""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        前向传播
        Args:
            input_ids: [batch_size, seq_length]
            token_type_ids: [batch_size, seq_length]
            position_ids: [batch_size, seq_length]
        """
        seq_length = input_ids.size(1)
        batch_size = input_ids.size(0)

        # 生成位置IDs
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # 生成段落类型IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 计算三种嵌入
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 合并嵌入并应用层归一化
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


import torch
import torch.nn as nn
import math


class BertSelfAttention(nn.Module):
    """BERT自注意力机制"""

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """转置张量以准备注意力计算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        """自注意力前向传播"""
        # 线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 转置以准备多头注意力
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 应用softmax和dropout
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(nn.Module):
    """BERT自注意力输出层"""

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """完整的BERT注意力模块"""

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


import torch
import torch.nn as nn


class BertIntermediate(nn.Module):
    """BERT中间层（前馈网络第一部分）"""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT输出层（前馈网络第二部分）"""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """完整的BERT层：注意力 + 前馈网络"""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力
        attention_output = self.attention(hidden_states, attention_mask)
        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


import torch
import torch.nn as nn


class BertEncoder(nn.Module):
    """BERT编码器：多个BERT层的堆叠"""

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(nn.Module):
    """BERT池化层：提取[CLS]标记的表示"""

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 取第一个token ([CLS]) 的表示
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """完整的BERT模型"""

    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 生成注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 扩展注意力掩码维度以匹配注意力分数
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 嵌入层
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # 编码器
        encoder_output = self.encoder(embedding_output, extended_attention_mask)

        # 池化层
        pooled_output = self.pooler(encoder_output)

        return {
            'last_hidden_state': encoder_output,
            'pooler_output': pooled_output
        }


import torch
import torch.nn as nn
from bert_config import BertConfig
from bert_model import BertModel
import numpy as np


class SimpleTokenizer:
    """简单的分词器用于演示"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

    def tokenize(self, text):
        """简单的分词方法"""
        words = text.lower().split()
        token_ids = []

        for word in words:
            # 简单的哈希函数将单词映射到token ID
            token_id = hash(word) % (self.vocab_size - 3) + 3  # 保留前3个给特殊token
            token_ids.append(token_id)

        return token_ids


def demonstrate_bert_model():
    """演示BERT模型的功能"""
    print("=== PyTorch BERT从零实现演示 ===\n")

    # 创建BERT配置
    config = BertConfig.from_pretrained("bert-base-uncased")

    # 创建BERT模型
    bert_model = BertModel(config)

    print("✓ BERT模型创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in bert_model.parameters()):,}")

    # 显示模型结构
    print("\n模型结构:")
    for name, module in bert_model.named_children():
        print(f"  - {name}")

    # 演示前向传播
    print("\n=== 前向传播演示 ===")

    # 创建模拟输入
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # 前向传播
    with torch.no_grad():
        outputs = bert_model(input_ids)

    print(f"输入形状: {input_ids.shape}")
    print(f"最后隐藏层状态形状: {outputs['last_hidden_state'].shape}")
    print(f"池化输出形状: {outputs['pooler_output'].shape}")

    # 显示配置信息
    print("\n=== 模型配置信息 ===")
    print(f"词汇表大小: {config.vocab_size}")
    print(f"隐藏层大小: {config.hidden_size}")
    print(f"注意力头数量: {config.num_attention_heads}")
    print(f"隐藏层层数: {config.num_hidden_layers}")
    print(f"中间层大小: {config.intermediate_size}")
    print(f"最大位置嵌入: {config.max_position_embeddings}")

    return bert_model


def test_model_components():
    """测试模型各个组件"""
    print("\n=== 模型组件测试 ===")

    config = BertConfig.from_pretrained("bert-base-uncased")

    # 测试嵌入层
    embeddings = BertEmbeddings(config)
    print(f"嵌入层参数数量: {sum(p.numel() for p in embeddings.parameters()):,}")

    # 测试注意力层
    attention = BertAttention(config)
    print(f"注意力层参数数量: {sum(p.numel() for p in attention.parameters()):,}")

    # 测试完整的BERT层
    bert_layer = BertLayer(config)
    print(f"BERT层参数数量: {sum(p.numel() for p in bert_layer.parameters()):,}")


def main():
    """主函数"""
    try:
        # 检查设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 演示BERT模型
        bert_model = demonstrate_bert_model()

        # 测试模型组件
        test_model_components()

        print(f"\n🎉 BERT模型从零实现完成!")
        print("模型包含:")
        print("  - BERT嵌入层（词嵌入+位置嵌入+段落嵌入）")
        print("  - 多头自注意力机制")
        print("  - 前馈神经网络")
        print("  - 层归一化和残差连接")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保所有依赖已正确安装")


if __name__ == "__main__":
    main()
