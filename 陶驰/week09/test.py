import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6
}
id2label = {v: k for k, v in label2id.items()}

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = list(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[labels[word_id]])
            else:
                label_ids.append(label2id[labels[word_id]])
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids)
        }

class BertForNER(nn.Module):
    def __init__(self, bert_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss, logits

texts = [
    "可以改成本地的text",
    "可以改成本地的text"
]

labels = [
    ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"],
    ["B-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "I-ORG"]
]

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
dataset = NERDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BertForNER("bert-base-chinese", num_labels=len(label2id))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_tensor = batch["labels"].to(device)

        loss, _ = model(input_ids, attention_mask, labels_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()
text = "可以改成本地的text"
encoding = tokenizer(
    list(text),
    is_split_into_words=True,
    return_tensors="pt"
).to(device)

_, logits = model(encoding["input_ids"], encoding["attention_mask"])
preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

word_ids = encoding.word_ids(batch_index=0)
result = []
prev = None
for idx, word_id in enumerate(word_ids):
    if word_id is None or word_id == prev:
        continue
    result.append((text[word_id], id2label[preds[idx]]))
    prev = word_id

print(result)
