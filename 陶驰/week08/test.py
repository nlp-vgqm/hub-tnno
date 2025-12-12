import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class TripletTextDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_len=64):
        self.data = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        a, p, n = self.data[idx]
        return (
            self.encode(a),
            self.encode(p),
            self.encode(n),
        )

    def __len__(self):
        return len(self.data)
class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-chinese", embedding_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, batch):
        outputs = self.bert(
            input_ids=batch["input_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1)
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls)

def train(model, dataloader, device):
    loss_fn = nn.TripletMarginLoss(margin=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for batch in dataloader:
        anchor, pos, neg = batch
        anchor = {k: v.to(device) for k, v in anchor.items()}
        pos = {k: v.to(device) for k, v in pos.items()}
        neg = {k: v.to(device) for k, v in neg.items()}

        a = model(anchor)
        p = model(pos)
        n = model(neg)

        loss = loss_fn(a, p, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss:", loss.item())


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    triplets = [
        ("今天天气很好", "天气真不错", "我在吃西瓜"),
        ("你喜欢看电影吗", "你爱看电影吧", "我今天打游戏"),
    ]

    dataset = TripletTextDataset(triplets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextEncoder().to(device)

    train(model, dataloader, device)
