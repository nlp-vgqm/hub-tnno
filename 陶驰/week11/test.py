import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SFTDataset(Dataset):
    def __init__(self, sources, targets, tokenizer, max_len=128):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.sources[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        tgt = self.tokenizer(
            self.targets[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = tgt["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "encoder_input_ids": src["input_ids"].squeeze(0),
            "encoder_attention_mask": src["attention_mask"].squeeze(0),
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": tgt["attention_mask"].squeeze(0),
            "labels": labels
        }

class BertSeq2Seq(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)

        decoder_config = BertConfig.from_pretrained(
            model_name,
            is_decoder=True,
            add_cross_attention=True
        )
        self.decoder = BertModel.from_pretrained(model_name, config=decoder_config)
        self.lm_head = nn.Linear(decoder_config.hidden_size, decoder_config.vocab_size, bias=False)

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None
    ):
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss, logits

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

sources = [
    "解释什么是机器学习",
    "用一句话介绍北京",
    "什么是监督学习"
]

targets = [
    "机器学习是让计算机从数据中学习规律的方法。",
    "北京是中国的首都。",
    "监督学习是使用带标签数据进行训练的方法。"
]

dataset = SFTDataset(sources, targets, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BertSeq2Seq("bert-base-chinese").to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)

model.train()
for epoch in range(3):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _ = model(**batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.eval()

src = "介绍一下机器学习"
enc = tokenizer(src, return_tensors="pt").to(device)

decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)

with torch.no_grad():
    for _ in range(30):
        _, logits = model(
            encoder_input_ids=enc["input_ids"],
            encoder_attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=torch.ones_like(decoder_input_ids)
        )
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=1)
        if next_id.item() == tokenizer.sep_token_id:
            break

print(tokenizer.decode(decoder_input_ids.squeeze(0)))
