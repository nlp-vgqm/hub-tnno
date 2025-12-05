import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Dataset ===================
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64, bert=False):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bert = bert

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.bert:
            encode = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                                    max_length=self.max_len, return_tensors="pt")
            return encode["input_ids"].squeeze(), encode["attention_mask"].squeeze(), torch.tensor(self.labels[idx])
        else:
            tokens = self.tokenizer(self.texts[idx])[:self.max_len]
            ids = tokens + [0]*(self.max_len-len(tokens))
            return torch.tensor(ids), torch.tensor(self.labels[idx])

#================= TextCNN ====================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, 128, k) for k in [2,3,4]])
        self.fc = nn.Linear(128*3, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        conv_out = [torch.relu(c(x)).max(dim=2)[0] for c in self.convs]
        out = torch.cat(conv_out, 1)
        return self.fc(out)

#================ LSTM ========================
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(out)

#================ BERT ========================
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, mask):
        cls = self.bert(input_ids, attention_mask=mask).last_hidden_state[:,0]
        return self.fc(cls)

#================ Train & Eval =================
def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        if len(batch)==3:
            ids, mask, labels = [x.to(device) for x in batch]
            out = model(ids, mask)
        else:
            x, labels = [x.to(device) for x in batch]
            out = model(x)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

def test(model, loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch)==3:
                ids, mask, labels = [x.to(device) for x in batch]
                out = model(ids, mask)
            else:
                x, labels = [x.to(device) for x in batch]
                out = model(x)
            preds += out.argmax(1).cpu().tolist()
            labels_all += labels.cpu().tolist()
    return accuracy_score(labels_all, preds)

#================ Main ========================
if __name__ == "__main__":
    df = pd.read_csv("文本分类练习.csv")
    df["label"] = df["label"].map({"好评":1, "差评":0})

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    model_name = "cnn"  # cnn / lstm / bert

    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        train_ds = TextDataset(train_df, tokenizer, bert=True)
        val_ds = TextDataset(val_df, tokenizer, bert=True)
        model = BertClassifier().to(device)
    else:
        tokenizer = lambda x: [ord(c) % 20000 for c in x]
        vocab = 20000
        train_ds = TextDataset(train_df, tokenizer)
        val_ds = TextDataset(val_df, tokenizer)
        model = TextCNN(vocab, 128).to(device) if model_name=="cnn" else TextLSTM(vocab).to(device)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(5):
        train_epoch(model, train_loader, loss_fn, optimizer)
        acc = test(model, val_loader)
        print(f"Epoch {epoch} | Accuracy: {acc:.4f}")
