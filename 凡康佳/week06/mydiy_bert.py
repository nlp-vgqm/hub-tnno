import torch
import math
import numpy as np
from transformers import BertModel
import torch.nn as nn

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称


        

class DiyBert(nn.Module):
    #将预训练好的整个权重字典输入进来
    def __init__(self, state_dict,vocab_size,embedding_dim=768):
        super(DiyBert, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.segment_embeddings = nn.Embedding(2, embedding_dim)
        self.position_embeddings = nn.Embedding(512, embedding_dim)
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False  # BERT 是 post-LN，所以设为 False
        )

        # 堆叠 12 层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.transformer_encoder.eval()
        self.pooler_output_layer = nn.Linear(embedding_dim, embedding_dim)
        self.tanh = nn.Tanh()
        self.load_weight(state_dict)

    def load_weight(self,state_dict):
        self.token_embeddings.weight.data = state_dict['embeddings.word_embeddings.weight']
        self.segment_embeddings.weight.data = state_dict['embeddings.token_type_embeddings.weight']
        self.position_embeddings.weight.data = state_dict['embeddings.position_embeddings.weight']
        self.embedding_layer_norm.weight.data = state_dict['embeddings.LayerNorm.weight']
        self.embedding_layer_norm.bias.data = state_dict['embeddings.LayerNorm.bias']
        for i in range(12):
            prefix_bert = f'encoder.layer.{i}.'
            prefix_torch = f'layers.{i}.'

            # 获取 BERT 第 i 层的 state dict（带前缀）
            bert_layer_sd = {
                k[len(prefix_bert):]: v 
                for k, v in bert.state_dict().items() 
                if k.startswith(prefix_bert)
            }

            # 映射 self-attention QKV
            q_w = bert_layer_sd['attention.self.query.weight']
            k_w = bert_layer_sd['attention.self.key.weight']
            v_w = bert_layer_sd['attention.self.value.weight']
            q_b = bert_layer_sd['attention.self.query.bias']
            k_b = bert_layer_sd['attention.self.key.bias']
            v_b = bert_layer_sd['attention.self.value.bias']

            in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)
            in_proj_bias = torch.cat([q_b, k_b, v_b], dim=0)

            # 映射到 torch layer
            state_dict[f'{prefix_torch}self_attn.in_proj_weight'] = in_proj_weight
            state_dict[f'{prefix_torch}self_attn.in_proj_bias'] = in_proj_bias
            state_dict[f'{prefix_torch}self_attn.out_proj.weight'] = bert_layer_sd['attention.output.dense.weight']
            state_dict[f'{prefix_torch}self_attn.out_proj.bias'] = bert_layer_sd['attention.output.dense.bias']

            # FFN
            state_dict[f'{prefix_torch}linear1.weight'] = bert_layer_sd['intermediate.dense.weight']
            state_dict[f'{prefix_torch}linear1.bias'] = bert_layer_sd['intermediate.dense.bias']
            state_dict[f'{prefix_torch}linear2.weight'] = bert_layer_sd['output.dense.weight']
            state_dict[f'{prefix_torch}linear2.bias'] = bert_layer_sd['output.dense.bias']

            # LayerNorm (post-norm: attention output -> norm1; ffn output -> norm2)
            state_dict[f'{prefix_torch}norm1.weight'] = bert_layer_sd['attention.output.LayerNorm.weight']
            state_dict[f'{prefix_torch}norm1.bias'] = bert_layer_sd['attention.output.LayerNorm.bias']
            state_dict[f'{prefix_torch}norm2.weight'] = bert_layer_sd['output.LayerNorm.weight']
            state_dict[f'{prefix_torch}norm2.bias'] = bert_layer_sd['output.LayerNorm.bias']


        
        self.pooler_output_layer.weight.data = state_dict['pooler.dense.weight']
        self.pooler_output_layer.bias.data = state_dict['pooler.dense.bias']
    

    #最终输出
    def forward(self, x):
        # x=torch.LongTensor(x)
        token_embeddings = self.token_embeddings(x)
        segment_embeddings = self.segment_embeddings(torch.zeros(x.shape, dtype=torch.long))
        position_embeddings = self.position_embeddings(torch.arange(x.shape[1], dtype=torch.long))
        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.embedding_layer_norm(embeddings)

        attention_output=self.transformer_encoder(embeddings)

        pooler_output = self.pooler_output_layer(embeddings[:, 0, :])
        pooler_output = self.tanh(pooler_output)

        return attention_output, pooler_output

        

        
        

#自制
db = DiyBert(state_dict,4)
diy_sequence_output, diy_pooler_output = db.forward(torch_x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

# print(diy_pooler_output)
# print(torch_pooler_output)
