import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_len=None):
        b, t, h = x.shape
        q = self.q(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if valid_len is not None:
            mask = torch.arange(t, device=x.device).expand(b, t) >= valid_len.unsqueeze(1)
            mask = mask.unsqueeze(1).unsqueeze(1).expand(b, self.num_heads, t, t)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(b, t, h)
        out = self.o(context)
        out = self.dropout(out)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_size, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.addnorm1 = AddNorm(hidden_size, dropout)
        self.ffn = PositionWiseFFN(hidden_size, ffn_size, dropout)
        self.addnorm2 = AddNorm(hidden_size, dropout)

    def forward(self, x, valid_len=None):
        y = self.mha(x, valid_len)
        x = self.addnorm1(x, y)
        y = self.ffn(x)
        x = self.addnorm2(x, y)
        return x


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.seg_embed = nn.Embedding(2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, segments):
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.token_embed(tokens) + self.pos_embed(pos) + self.seg_embed(segments)
        x = self.dropout(x)
        return x


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, ffn_size=1024, num_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = BERTEmbedding(vocab_size, hidden_size, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_size, num_heads, ffn_size, dropout) for _ in range(num_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, tokens, segments, valid_len=None):
        x = self.embed(tokens, segments)
        for layer in self.layers:
            x = layer(x, valid_len)
        return x


class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, ffn_size=1024, num_layers=4, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = BERTEncoder(vocab_size, hidden_size, num_heads, ffn_size, num_layers, max_len, dropout)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pool_act = nn.Tanh()

    def forward(self, tokens, segments, valid_len=None):
        x = self.encoder(tokens, segments, valid_len)
        cls = x[:, 0]
        pooled = self.pool_act(self.pooler(cls))
        return x, pooled


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 1000
    model = BERTModel(vocab_size=vocab_size, hidden_size=128, num_heads=4, ffn_size=256, num_layers=2, max_len=64, dropout=0.1)
    model.eval()
    b, t = 2, 16
    tokens = torch.randint(0, vocab_size, (b, t))
    segments = torch.zeros(b, t, dtype=torch.long)
    valid_len = torch.tensor([t, t - 4])
    with torch.no_grad():
        seq_out, pooled = model(tokens, segments, valid_len)
    print(seq_out.shape, pooled.shape)
