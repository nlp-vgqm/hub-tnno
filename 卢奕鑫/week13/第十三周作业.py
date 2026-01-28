#!/usr/bin/env python3
"""
完全兼容的LoRA NER训练代码 - 修复版本
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 标签定义
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


class LoRALayer(nn.Module):
    """简化的LoRA层"""

    def __init__(self, base_layer, r=4, alpha=8, dropout=0.1):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        # 获取基础层的形状
        if hasattr(base_layer, 'weight'):
            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]

            # LoRA参数
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

            # 初始化
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)

            # 冻结基础层
            for param in base_layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_output + self.scaling * lora_output


class SimpleSelfAttention(nn.Module):
    """简化的自注意力机制"""

    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q, K, V投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 应用LoRA到Q, K, V
        self.query = LoRALayer(self.query, r=2, alpha=4)
        self.key = LoRALayer(self.key, r=2, alpha=4)
        self.value = LoRALayer(self.value, r=2, alpha=4)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.shape

        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 应用attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        context = torch.matmul(attention_weights, V)

        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        return self.out_proj(context)


class SimpleTransformerBlock(nn.Module):
    """简化的Transformer块"""

    def __init__(self, hidden_size, num_heads=4, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = hidden_size * 4

        self.attention = SimpleSelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
        )

        # 对FFN应用LoRA
        self.ffn[0] = LoRALayer(self.ffn[0], r=2, alpha=4)
        self.ffn[3] = LoRALayer(self.ffn[3], r=2, alpha=4)

    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接
        attn_output = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class SimpleBERTWithLoRA(nn.Module):
    """简化的BERT + LoRA模型"""

    def __init__(self, vocab_size=30522, num_labels=9, hidden_size=96, num_layers=2, num_heads=4):
        super().__init__()

        # 确保hidden_size能被num_heads整除
        assert hidden_size % num_heads == 0, f"hidden_size({hidden_size})必须能被num_heads({num_heads})整除"

        self.hidden_size = hidden_size

        # 嵌入层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # 分类头（应用LoRA）
        classifier = nn.Linear(hidden_size, num_labels)
        self.classifier = LoRALayer(classifier, r=4, alpha=8)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape

        # 词嵌入
        word_embeds = self.word_embeddings(input_ids)

        # 位置嵌入
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        # 合并嵌入
        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 通过Transformer层
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # 分类
        logits = self.classifier(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            # 重塑logits和labels
            active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else None

            if active_loss is not None:
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits) if loss is not None else logits


class SimpleTokenizer:
    """简化的Tokenizer"""

    def __init__(self):
        self.vocab = {}

        # 添加特殊token
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i

        # 添加常见单词
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "person", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        ]

        for i, word in enumerate(common_words, start=len(self.vocab)):
            self.vocab[word] = i

        # 添加大写字母
        for i in range(65, 91):  # A-Z
            self.vocab[chr(i)] = len(self.vocab)

        # 添加小写字母
        for i in range(97, 123):  # a-z
            self.vocab[chr(i)] = len(self.vocab)

        # 添加数字
        for i in range(10):  # 0-9
            self.vocab[str(i)] = len(self.vocab)

        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]
        self.cls_token_id = self.vocab["[CLS]"]
        self.sep_token_id = self.vocab["[SEP]"]

    def tokenize(self, text):
        """简单分词：按空格分割，保留标点"""
        tokens = []
        current_token = ""

        for char in text.lower():
            if char.isalnum():
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if char.strip():  # 如果不是空格
                    tokens.append(char)

        if current_token:
            tokens.append(current_token)

        return tokens

    def convert_tokens_to_ids(self, tokens):
        """token转id"""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def encode(self, text, max_length=128, add_special_tokens=True):
        """编码文本"""
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        input_ids = self.convert_tokens_to_ids(tokens)

        # 截断和填充
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length

        # attention mask
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask])
        }


class NERDataset(Dataset):
    """NER数据集"""

    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]

        # 将tokens组合成文本
        text = " ".join(tokens)

        # 编码文本
        encoding = self.tokenizer.encode(text, max_length=self.max_length, add_special_tokens=True)

        # 对齐标签
        labels = item["ner_tags"]
        aligned_labels = []

        # 简单对齐：为每个token分配标签
        # 由于我们使用了简单的分词，大多数token应该能对齐
        for i, token in enumerate(tokens):
            if i < len(labels):
                aligned_labels.append(labels[i])
            else:
                aligned_labels.append(-100)  # 填充

        # 添加特殊token的标签
        aligned_labels = [-100] + aligned_labels + [-100]  # CLS和SEP

        # 调整长度
        if len(aligned_labels) > self.max_length:
            aligned_labels = aligned_labels[:self.max_length]
        else:
            aligned_labels = aligned_labels + [-100] * (self.max_length - len(aligned_labels))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


def create_training_data():
    """创建训练数据"""
    data = [
        {
            "tokens": ["John", "Smith", "works", "at", "Google", "in", "London"],
            "ner_tags": [1, 2, 0, 0, 3, 0, 5]  # B-PER, I-PER, O, O, B-ORG, O, B-LOC
        },
        {
            "tokens": ["Microsoft", "was", "founded", "by", "Bill", "Gates"],
            "ner_tags": [3, 0, 0, 0, 1, 2]  # B-ORG, O, O, O, B-PER, I-PER
        },
        {
            "tokens": ["Apple", "CEO", "Tim", "Cook", "announced", "new", "products"],
            "ner_tags": [3, 0, 1, 2, 0, 0, 0]  # B-ORG, O, B-PER, I-PER, O, O, O
        },
        {
            "tokens": ["Amazon", "is", "based", "in", "Seattle", "USA"],
            "ner_tags": [3, 0, 0, 0, 5, 5]  # B-ORG, O, O, O, B-LOC, B-LOC
        },
        {
            "tokens": ["Elon", "Musk", "owns", "Tesla", "and", "SpaceX"],
            "ner_tags": [1, 2, 0, 3, 0, 3]  # B-PER, I-PER, O, B-ORG, O, B-ORG
        },
        {
            "tokens": ["Facebook", "now", "called", "Meta", "Platforms"],
            "ner_tags": [3, 0, 0, 3, 3]  # B-ORG, O, O, B-ORG, B-ORG
        },
        {
            "tokens": ["Barack", "Obama", "was", "born", "in", "Hawaii"],
            "ner_tags": [1, 2, 0, 0, 0, 5]  # B-PER, I-PER, O, O, O, B-LOC
        },
        {
            "tokens": ["The", "United", "Nations", "meeting", "in", "New", "York"],
            "ner_tags": [0, 3, 3, 0, 0, 5, 5]  # O, B-ORG, I-ORG, O, O, B-LOC, I-LOC
        },
        {
            "tokens": ["iPhone", "is", "made", "by", "Apple", "Inc"],
            "ner_tags": [0, 0, 0, 0, 3, 3]  # O, O, O, O, B-ORG, I-ORG
        },
        {
            "tokens": ["Mark", "Zuckerberg", "founded", "Facebook", "in", "2004"],
            "ner_tags": [1, 2, 0, 3, 0, 0]  # B-PER, I-PER, O, B-ORG, O, O
        }
    ]
    return data


def train_model():
    """训练模型"""
    print("1. 准备数据...")
    data = create_training_data()
    print(f"创建了 {len(data)} 条训练数据")

    print("2. 初始化tokenizer...")
    tokenizer = SimpleTokenizer()
    print(f"词汇表大小: {len(tokenizer.vocab)}")

    print("3. 创建数据集...")
    dataset = NERDataset(data, tokenizer, max_length=32)

    print("4. 创建模型...")
    # 使用能整除的参数
    model = SimpleBERTWithLoRA(
        vocab_size=len(tokenizer.vocab),
        num_labels=len(label_list),
        hidden_size=96,  # 96能被4整除
        num_layers=2,
        num_heads=4
    )

    # 打印参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"训练参数量占比: {100 * trainable_params / total_params:.2f}%")

    print("5. 设置训练参数...")
    batch_size = 2
    epochs = 20
    learning_rate = 3e-4

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 如果有GPU，使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)

    print("6. 开始训练...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}完成，平均损失: {avg_loss:.4f}")

        # 每5个epoch保存一次
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

    print("\n训练完成!")
    return model, tokenizer


def predict(model, tokenizer, text):
    """预测函数"""
    model.eval()
    device = next(model.parameters()).device

    # 编码文本
    encoding = tokenizer.encode(text, max_length=32, add_special_tokens=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 预测
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)

    # 解码token
    tokens = tokenizer.tokenize(text)

    # 获取预测的标签
    pred_labels = predictions.squeeze().cpu().numpy()

    # 只取有效部分（跳过CLS和SEP，考虑padding）
    valid_indices = attention_mask.squeeze().cpu().numpy() == 1
    pred_labels = pred_labels[valid_indices]

    # 跳过第一个CLS token
    if len(pred_labels) > 0:
        pred_labels = pred_labels[1:]

    # 打印结果
    print(f"\n文本: {text}")
    print("-" * 50)

    current_entity = []
    current_type = None

    for i, (token, label_id) in enumerate(zip(tokens, pred_labels[:len(tokens)])):
        label = id2label.get(label_id, "O")

        if label.startswith("B-"):
            # 输出之前的实体
            if current_entity:
                print(f"实体: {' '.join(current_entity)}")
                print(f"类型: {current_type}")
                print("-" * 30)

            # 开始新实体
            current_entity = [token]
            current_type = label[2:]

        elif label.startswith("I-") and current_type and label[2:] == current_type:
            # 继续当前实体
            current_entity.append(token)

        else:
            # 输出之前的实体
            if current_entity:
                print(f"实体: {' '.join(current_entity)}")
                print(f"类型: {current_type}")
                print("-" * 30)
                current_entity = []
                current_type = None

    # 输出最后一个实体
    if current_entity:
        print(f"实体: {' '.join(current_entity)}")
        print(f"类型: {current_type}")
        print("-" * 30)


def save_model(model, tokenizer, path="./saved_model"):
    """保存模型"""
    os.makedirs(path, exist_ok=True)

    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pth"))

    # 保存tokenizer词汇表
    with open(os.path.join(path, "vocab.json"), "w") as f:
        json.dump(tokenizer.vocab, f, indent=2)

    # 保存标签映射
    with open(os.path.join(path, "labels.json"), "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": id2label,
            "label_list": label_list
        }, f, indent=2)

    # 保存模型配置
    config = {
        "vocab_size": len(tokenizer.vocab),
        "num_labels": len(label_list),
        "hidden_size": model.hidden_size,
        "num_layers": len(model.transformer_layers),
        "num_heads": 4  # 硬编码，因为模型结构如此
    }

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n模型已保存到: {path}")


def main():
    """主函数"""
    print("=" * 60)
    print("LoRA NER 训练程序 - 最终版")
    print("=" * 60)

    try:
        # 训练模型
        model, tokenizer = train_model()

        # 保存模型
        save_model(model, tokenizer)

        # 测试预测
        print("\n" + "=" * 60)
        print("测试预测结果:")
        print("=" * 60)

        test_texts = [
            "John Smith works at Google in London",
            "Microsoft was founded by Bill Gates",
            "Apple CEO Tim Cook announced new iPhone",
            "Elon Musk owns Tesla and SpaceX",
            "Mark Zuckerberg founded Facebook",
            "Barack Obama was born in Hawaii"
        ]

        for text in test_texts:
            predict(model, tokenizer, text)

        print("\n" + "=" * 60)
        print("程序运行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
