import json
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np


def create_sft_attention_mask(input_ids, attention_mask, tokenizer, max_length):
    """
    创建SFT注意力掩码（在数据处理阶段生成）

    Args:
        input_ids: 原始输入ID
        attention_mask: 原始注意力掩码
        tokenizer: tokenizer对象
        max_length: 最大长度

    Returns:
        sft_attention_mask: seq_len * seq_len 的注意力掩码
    """
    seq_len = len(input_ids)
    sft_mask = torch.ones(seq_len, seq_len, dtype=torch.float32)

    # 找到SEP的位置
    try:
        sep_idx = input_ids.index(tokenizer.sep_token_id)
    except ValueError:
        # 如果没有找到SEP，则假设整个序列都是问题
        sep_idx = seq_len - 1

    # 问题部分可以看到整个问题
    for i in range(sep_idx + 1):
        for j in range(seq_len):
            if j > sep_idx:  # 问题不能看到回答
                sft_mask[i, j] = 0

    # 回答部分只能看到问题和前面的回答（因果注意力）
    for i in range(sep_idx + 1, seq_len):
        for j in range(seq_len):
            if j > i:  # 不能看到未来的token
                sft_mask[i, j] = 0

    # 处理padding
    for i in range(seq_len):
        if attention_mask[i] == 0:  # padding位置
            sft_mask[i, :] = 0
            sft_mask[:, i] = 0

    return sft_mask


class BertSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, split_type="train", split_ratio=(0.7, 0.15, 0.15),
                 seed=42):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            tokenizer: tokenizer对象
            max_length: 最大长度
            split_type: 数据集类型 ("train", "val", "test")
            split_ratio: 训练集、验证集、测试集比例
            seed: 随机种子
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.split_type = split_type
        self.load_and_split_data(data_path, split_ratio, seed)

    def load_and_split_data(self, path, split_ratio, seed):
        """加载并划分数据"""
        all_samples = []
        with open(path, encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                target = item["target"]
                for question in item["questions"]:
                    all_samples.append((question, target))

        # 设置随机种子
        random.seed(seed)
        torch.manual_seed(seed)

        # 划分数据集
        train_ratio, val_ratio, test_ratio = split_ratio
        n_total = len(all_samples)

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # 随机打乱
        random.shuffle(all_samples)

        if self.split_type == "train":
            self.samples = all_samples[:n_train]
        elif self.split_type == "val":
            self.samples = all_samples[n_train:n_train + n_val]
        elif self.split_type == "test":
            self.samples = all_samples[n_train + n_val:]

        print(f"{self.split_type.upper()} dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        question, target = self.samples[index]

        # 构造输入：[CLS] 问题 [SEP] 答案 [SEP]
        input_str = f"{question}{self.tokenizer.sep_token}{target}"

        # 编码
        encoding = self.tokenizer(
            input_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 构造Labels
        sep_idx = input_ids.index(self.tokenizer.sep_token_id)
        labels = [-100] * (sep_idx + 1)  # Question和第一个SEP不算Loss
        target_ids = input_ids[sep_idx + 1:]  # Target部分
        labels.extend(target_ids)

        # 填充到max_length
        labels = labels[:self.max_length]
        if len(labels) < self.max_length:
            labels += [-100] * (self.max_length - len(labels))

        # 创建SFT注意力掩码
        sft_attention_mask = create_sft_attention_mask(
            input_ids,
            attention_mask,
            self.tokenizer,
            self.max_length
        )

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "sft_attention_mask": sft_attention_mask,
            "labels": torch.LongTensor(labels),
            "question": question,  # 保存原始问题用于测试输出
            "target": target  # 保存原始答案用于测试输出
        }


class BertForSFT(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.config = self.bert.config
        self.cls_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, sft_attention_mask=None, labels=None):
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            sft_attention_mask: 预先生成的SFT注意力掩码
            labels: 标签
        """
        if sft_attention_mask is None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            sft_attention_mask = torch.ones(batch_size, seq_len, seq_len, device=device)

        outputs = self.bert(input_ids=input_ids, attention_mask=sft_attention_mask)
        sequence_output = outputs[0]
        logits = self.cls_head(sequence_output)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def collate_fn(batch):
    """自定义collate函数处理批量数据"""
    batch_size = len(batch)
    max_len = batch[0]["input_ids"].shape[0]

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    sft_attention_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    questions = []
    targets = []

    for i, sample in enumerate(batch):
        input_ids[i] = sample["input_ids"]
        attention_mask[i] = sample["attention_mask"]
        sft_attention_mask[i] = sample["sft_attention_mask"]
        labels[i] = sample["labels"]
        questions.append(sample.get("question", ""))
        targets.append(sample.get("target", ""))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "sft_attention_mask": sft_attention_mask,
        "labels": labels,
        "questions": questions,
        "targets": targets
    }


def evaluate(model, dataloader, device, tokenizer, output_file=None):
    """评估模型并在测试集上输出结果"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_questions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            sft_attention_mask = batch["sft_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            questions = batch["questions"]
            targets = batch["targets"]

            outputs = model(input_ids, sft_attention_mask, labels)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()

            # 生成预测
            predictions = generate_predictions(model, input_ids, sft_attention_mask, tokenizer, device)

            for i in range(len(questions)):
                all_predictions.append(predictions[i])
                all_questions.append(questions[i])
                all_targets.append(targets[i])

    avg_loss = total_loss / len(dataloader)

    # 输出结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Test Loss: {avg_loss:.4f}\n")
            f.write("=" * 80 + "\n")

            for i, (question, target, prediction) in enumerate(zip(all_questions, all_targets, all_predictions)):
                f.write(f"Sample {i + 1}:\n")
                f.write(f"Question: {question}\n")
                f.write(f"Target: {target}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write("-" * 40 + "\n")

        print(f"Results saved to {output_file}")

    # 控制台输出
    print(f"Test Loss: {avg_loss:.4f}")
    print("\nSample Predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"\nSample {i + 1}:")
        print(f"Question: {all_questions[i]}")
        print(f"Target: {all_targets[i]}")
        print(f"Prediction: {all_predictions[i]}")
        print("-" * 40)

    return avg_loss


def generate_predictions(model, input_ids, attention_mask, tokenizer, device, max_new_tokens=50):
    """生成预测文本"""
    model.eval()
    batch_size = input_ids.shape[0]
    predictions = []

    with torch.no_grad():
        for i in range(batch_size):
            # 找到SEP的位置
            sep_idx = torch.where(input_ids[i] == tokenizer.sep_token_id)[0]
            if len(sep_idx) > 0:
                start_idx = sep_idx[0].item() + 1
            else:
                start_idx = 0

            # 初始化生成序列
            generated = input_ids[i].clone().unsqueeze(0)

            for _ in range(max_new_tokens):
                # 获取模型输出
                output = model(generated.to(device), attention_mask[i].unsqueeze(0).to(device))
                next_token_logits = output["logits"][:, -1, :]

                # 使用贪心解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 如果生成了[SEP]或[PAD]，则停止
                if next_token.item() == tokenizer.sep_token_id or next_token.item() == tokenizer.pad_token_id:
                    break

                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)

                # 更新attention_mask
                if attention_mask.shape[-1] > generated.shape[1]:
                    new_attention = attention_mask[i, :generated.shape[1]].unsqueeze(0)
                else:
                    break

            # 解码生成的结果
            generated_tokens = generated[0, start_idx:].tolist()
            # 移除[PAD]和[SEP] token
            filtered_tokens = [t for t in generated_tokens if t not in [tokenizer.pad_token_id, tokenizer.sep_token_id]]
            prediction = tokenizer.decode(filtered_tokens, skip_special_tokens=True)
            predictions.append(prediction)

    return predictions


def train_and_evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = r"D:\就业\LLM\第六周 语言模型\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 创建训练、验证、测试数据集
    print("\nLoading datasets...")
    train_dataset = BertSFTDataset("data.json", tokenizer, split_type="train")
    val_dataset = BertSFTDataset("data.json", tokenizer, split_type="val")
    test_dataset = BertSFTDataset("data.json", tokenizer, split_type="test")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    model = BertForSFT(model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 训练模型
    print("\nStarting training...")
    best_val_loss = float('inf')
    for epoch in range(5):
        # 训练阶段
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            sft_attention_mask = batch["sft_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, sft_attention_mask, labels)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                sft_attention_mask = batch["sft_attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, sft_attention_mask, labels)
                val_loss += outputs["loss"].item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch} completed:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Best model saved with validation loss: {best_val_loss:.4f}")

    # 在测试集上评估
    print("\nEvaluating on test set...")
    test_loss = evaluate(model, test_loader, device, tokenizer, output_file="test_results.txt")
    print(f"\nFinal Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
