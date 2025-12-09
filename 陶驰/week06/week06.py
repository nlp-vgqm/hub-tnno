import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(dropout_prob)
        self.proj_dropout = nn.Dropout(dropout_prob)

        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(dropout_prob)

    def _shape_qkv(self, x):
        b, s, _ = x.size()
        x = x.view(b, s, self.num_heads, 3 * self.head_dim)
        x = x.permute(2, 0, 1, 3)
        q, k, v = torch.split(x, self.head_dim, dim=-1)
        q = q.permute(1,0,2,3)
        k = k.permute(1,0,2,3)
        v = v.permute(1,0,2,3)
        return q, k, v

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        b, seq_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states)
        q, k, v = self._shape_qkv(qkv)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                attention_mask = (1.0 - attention_mask) * -1e9
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores + attention_mask

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0,2,1,3).contiguous().view(b, seq_len, self.hidden_size)

        attn_output = self.out_proj(context)
        attn_output = self.proj_dropout(attn_output)

        hidden_states = self.attn_layer_norm(hidden_states + attn_output)

        hidden_intermediate = self.intermediate(hidden_states)
        hidden_intermediate = self.intermediate_act_fn(hidden_intermediate)
        hidden_intermediate = self.output_dense(hidden_intermediate)
        hidden_intermediate = self.output_dropout(hidden_intermediate)

        layer_output = self.ffn_layer_norm(hidden_states + hidden_intermediate)

        return layer_output

if __name__ == "__main__":
    model = TransformerLayer(hidden_size=768, num_attention_heads=12, intermediate_size=3072)
    dummy = torch.randn(2, 8, 768)
    mask = torch.ones(2, 8)
    out = model(dummy, mask)
    print(out.shape)
