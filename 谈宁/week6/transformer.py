import torch
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained(model_path, return_dict=False)
bert.eval()
state_dict = bert.state_dict()

torch_x = torch.LongTensor([x])

with torch.no_grad():
    torch_sequence_output, _ = bert(torch_x)


class MyBert:
    def __init__(self, state_dict):
        self.w = state_dict
        self.hidden = 768
        self.heads = 12
        self.head_dim = 64
    def embedding(self, x):
        we = F.embedding(x, self.get_w("embeddings.word_embeddings.weight"))
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pe = F.embedding(pos, self.get_w("embeddings.position_embeddings.weight"))
        te = F.embedding(torch.zeros_like(x), self.get_w("embeddings.token_type_embeddings.weight"))
        return self.ln(we + pe + te, "embeddings.LayerNorm")

    def attention(self, x, i):
        pre = f"encoder.layer.{i}.attention"
        q = F.linear(x, self.get_w(f"{pre}.self.query.weight"), self.get_w(f"{pre}.self.query.bias"))
        k = F.linear(x, self.get_w(f"{pre}.self.key.weight"), self.get_w(f"{pre}.self.key.bias"))
        v = F.linear(x, self.get_w(f"{pre}.self.value.weight"), self.get_w(f"{pre}.self.value.bias"))

        def split(t):
            return t.view(1, -1, 12, 64).permute(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(64)
        probs = F.softmax(score, dim=-1)
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(1, -1, 768)

        out = F.linear(context, self.get_w(f"{pre}.output.dense.weight"), self.get_w(f"{pre}.output.dense.bias"))
        return self.ln(x + out, f"{pre}.output.LayerNorm")

    def feed_forward(self, x, i):
        pre = f"encoder.layer.{i}"
        mid = F.linear(x, self.get_w(f"{pre}.intermediate.dense.weight"), self.get_w(f"{pre}.intermediate.dense.bias"))
        mid = F.gelu(mid)
        out = F.linear(mid, self.get_w(f"{pre}.output.dense.weight"), self.get_w(f"{pre}.output.dense.bias"))
        return self.ln(x + out, f"{pre}.output.LayerNorm")
    def get_w(self, name):
        return self.w[name]

    def ln(self, x, name):
        w = self.get_w(f"{name}.weight")
        b = self.get_w(f"{name}.bias")
        return F.layer_norm(x, (768,), weight=w, bias=b, eps=1e-12)



    def forward(self, x):
        out = self.embedding(x)
        for i in range(12):
            out = self.attention(out, i)
            out = self.feed_forward(out, i)
        return out


my_bert = MyBert(state_dict)
with torch.no_grad():
    diy_sequence_output = my_bert.forward(torch_x)
