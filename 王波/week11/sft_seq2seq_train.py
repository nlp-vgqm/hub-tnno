import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random
import re
from tqdm import tqdm
import json

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 超参数配置 - 增加模型容量
class Config:
    batch_size = 32
    embedding_dim = 256  # 增加嵌入维度
    hidden_dim = 512  # 增加隐藏层维度
    max_len = 25
    epochs = 50
    learning_rate = 0.001
    dropout = 0.2
    teacher_forcing_ratio = 0.5
    vocab_size = 5000  # 增加词汇表大小
    min_freq = 2  # 最低词频


config = Config()


# ========== 1. 增强版数据处理 ==========
class EnhancedTextProcessor:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = Counter()

    def preprocess(self, text):
        """改进的文本预处理"""
        text = str(text).lower().strip()
        # 保留更多标点符号
        text = re.sub(r"[^a-z0-9?.!,':\s]+", " ", text)
        # 规范化空格
        text = re.sub(r'\s+', ' ', text)
        return text

    def build_vocab(self, texts, max_vocab_size=5000, min_freq=2):
        """构建更大的词汇表"""
        print("正在构建词汇表...")

        # 统计所有文本中的词频
        for text in texts:
            words = text.split()
            self.word_count.update(words)

        print(f"发现总词汇数: {len(self.word_count)}")

        # 过滤低频词
        filtered_words = {word: count for word, count in self.word_count.items()
                          if count >= min_freq}

        # 按频率排序，选择最常见的词
        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        # 限制词汇表大小，但保留所有高频词
        num_to_keep = min(max_vocab_size - 4, len(sorted_words))

        for idx, (word, _) in enumerate(sorted_words[:num_to_keep], start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"词汇表大小: {len(self.word2idx)}")
        print(f"最常用的10个词: {sorted_words[:10]}")

        # 计算覆盖率
        total_words = sum(self.word_count.values())
        covered_words = sum(count for word, count in self.word_count.items()
                            if word in self.word2idx)
        coverage = covered_words / total_words * 100
        print(f"词汇表覆盖率: {coverage:.2f}%")

    def encode(self, text, max_len=None):
        """改进的编码函数"""
        words = text.split()

        # 转换词到索引，未知词用<UNK>
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]

        # 添加特殊标记
        indices = [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]

        # 处理长度
        if max_len:
            if len(indices) < max_len:
                # 填充
                indices = indices + [self.word2idx['<PAD>']] * (max_len - len(indices))
            else:
                # 截断但确保以EOS结尾
                indices = indices[:max_len - 1] + [self.word2idx['<EOS>']]

        return indices

    def decode(self, indices):
        """改进的解码函数"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                word = self.idx2word.get(idx, '<UNK>')
                words.append(word)
        return ' '.join(words)


# ========== 2. 加载1000条数据集 ==========
def load_dataset_from_file(filename='train_pairs_1000.txt'):
    """从文件加载数据集"""
    input_texts = []
    output_texts = []

    print(f"正在加载数据集: {filename}")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line and '\t' in line:
            input_text, output_text = line.split('\t', 1)
            input_texts.append(input_text.strip())
            output_texts.append(output_text.strip())

    print(f"成功加载 {len(input_texts)} 条对话")
    return input_texts[:1000], output_texts[:1000]  # 确保只取1000条


def create_synthetic_dataset():
    """如果文件不存在，创建合成数据集"""
    print("创建合成数据集...")

    # 基础对话模式
    pairs = []

    # 添加更丰富的对话
    base_pairs = [
        ("hello", "hi there!"),
        ("hi", "hello!"),
        ("how are you", "i'm good, thanks!"),
        ("what is your name", "my name is chat assistant"),
        ("good morning", "good morning to you too"),
        ("goodbye", "see you later"),
        ("thank you", "you're welcome"),
        ("what time is it", "it's time to learn"),
        ("do you like ai", "yes i love artificial intelligence"),
        ("where are you from", "i'm from the digital world"),
        ("how old are you", "i don't have an age"),
        ("what is ai", "artificial intelligence is the future"),
        ("can you help me", "yes i can help with many things"),
        ("are you human", "no i am an ai assistant"),
        ("do you sleep", "no i never sleep"),
        ("what can you do", "i can answer questions and chat"),
        ("who made you", "i was created by developers"),
        ("what is your purpose", "to help and assist people"),
        ("are you smart", "i try to be helpful"),
        ("i am happy", "that's great to hear"),
    ]

    # 生成更多变体
    for input_text, output_text in base_pairs:
        pairs.append((input_text, output_text))

        # 添加变体
        pairs.append((input_text.upper(), output_text))
        pairs.append((input_text.capitalize(), output_text))
        pairs.append((f"hey, {input_text}", output_text))
        pairs.append((f"so, {input_text}", output_text))

    # 添加更多对话
    more_pairs = [
        ("what's the weather", "it's sunny and warm"),
        ("tell me a joke", "why did the chicken cross the road"),
        ("i love pizza", "pizza is delicious"),
        ("my name is john", "nice to meet you john"),
        ("i'm from new york", "new york is a great city"),
        ("i work as a teacher", "teaching is an important job"),
        ("i like to read books", "reading is a good habit"),
        ("what is python", "python is a programming language"),
        ("how to learn programming", "start with basics and practice daily"),
        ("what is machine learning", "ml is a subset of artificial intelligence"),
    ]

    pairs.extend(more_pairs)

    # 重复生成直到有1000条
    while len(pairs) < 1000:
        base_input, base_output = random.choice(base_pairs)
        # 轻微修改
        new_input = base_input
        if random.random() > 0.5:
            new_input = f"so, {base_input}"
        pairs.append((new_input, base_output))

    # 分割为输入和输出
    input_texts = [p[0] for p in pairs[:1000]]
    output_texts = [p[1] for p in pairs[:1000]]

    return input_texts, output_texts


# ========== 3. 数据集类 ==========
class ConversationDataset(Dataset):
    def __init__(self, input_texts, output_texts, processor, max_len=25):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.processor = processor
        self.max_len = max_len

        # 构建词汇表
        all_texts = input_texts + output_texts
        self.processor.build_vocab(all_texts, max_vocab_size=config.vocab_size,
                                   min_freq=config.min_freq)

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        # 预处理
        input_text = self.processor.preprocess(input_text)
        output_text = self.processor.preprocess(output_text)

        input_ids = self.processor.encode(input_text, self.max_len)
        output_ids = self.processor.encode(output_text, self.max_len)

        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'output': torch.tensor(output_ids, dtype=torch.long),
            'input_text': input_text,
            'output_text': output_text
        }


# ========== 4. 模型组件（保持原样但增加容量）==========
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden_repeated = hidden.transpose(0, 1)
        hidden_repeated = hidden_repeated.repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.fc(output.squeeze(1))

        return output, hidden, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


# ========== 5. 训练函数 ==========
def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0

    for batch in tqdm(dataloader, desc="训练"):
        src = batch['input'].to(device)
        trg = batch['output'].to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def generate_response(model, processor, input_text, max_len=25):
    model.eval()

    input_text = processor.preprocess(input_text)
    input_ids = processor.encode(input_text, max_len)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(input_tensor)
        trg_ids = [processor.word2idx['<SOS>']]

        for _ in range(max_len):
            input_token = torch.tensor([trg_ids[-1]]).to(device)
            output, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            pred_token = output.argmax().item()
            trg_ids.append(pred_token)
            if pred_token == processor.word2idx['<EOS>']:
                break

        response = processor.decode(trg_ids)

    return response


# ========== 6. 主训练流程 ==========
def main():
    print("开始训练增强版Seq2Seq模型...")
    print("=" * 60)

    # 尝试从文件加载数据集，如果不存在则创建
    try:
        input_texts, output_texts = load_dataset_from_file('seq2seq_data_1000_20260108_215649.txtwhere')
        print(f"从文件加载了 {len(input_texts)} 条对话")
    except FileNotFoundError:
        print("未找到数据集文件，创建合成数据集...")
        input_texts, output_texts = create_synthetic_dataset()
        # 保存生成的数据集
        with open('synthetic_dataset.txt', 'w', encoding='utf-8') as f:
            for inp, out in zip(input_texts, output_texts):
                f.write(f"{inp}\t{out}\n")
        print(f"创建了 {len(input_texts)} 条合成对话")

    print(f"样本示例:")
    for i in range(min(3, len(input_texts))):
        print(f"  输入: {input_texts[i]}")
        print(f"  输出: {output_texts[i]}")
        print()

    # 创建处理器
    processor = EnhancedTextProcessor()

    # 创建数据集
    dataset = ConversationDataset(input_texts, output_texts, processor, config.max_len)
    vocab_size = len(processor.word2idx)
    print(f"最终词汇表大小: {vocab_size}")

    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")

    # 创建模型
    encoder = Encoder(vocab_size, config.embedding_dim, config.hidden_dim)
    decoder = Decoder(vocab_size, config.embedding_dim, config.hidden_dim)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 config.teacher_forcing_ratio)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['input'].to(device)
                trg = batch['output'].to(device)
                output = model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 学习率调整
        scheduler.step(val_loss)

        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'processor': processor,
                'word2idx': processor.word2idx,
                'idx2word': processor.idx2word,
                'config': config.__dict__
            }, 'best_seq2seq_model.pth')
            print(f"保存最佳模型到 'best_seq2seq_model.pth'")

        # 每5个epoch测试生成
        if (epoch + 1) % 5 == 0:
            print("\n测试生成:")
            test_inputs = ["hello", "how are you", "what is your name",
                           "tell me a joke", "what is ai"]

            for test_input in test_inputs:
                response = generate_response(model, processor, test_input)
                print(f"  输入: '{test_input}' -> 输出: '{response}'")

    print("\n训练完成!")

    # 最终测试
    print("\n最终测试:")
    model.eval()

    test_cases = [
        "hello",
        "hi there",
        "how are you today",
        "what is your name",
        "good morning",
        "thank you for your help",
        "what time is it",
        "do you like programming",
        "where are you from",
        "can you tell me a story"
    ]

    for test_input in test_cases:
        response = generate_response(model, processor, test_input)
        print(f"输入: {test_input}")
        print(f"输出: {response}")
        print("-" * 40)

    # 交互模式
    print("\n进入交互模式 (输入 'quit' 退出):")
    model.eval()

    while True:
        try:
            user_input = input("\n你: ").strip()

            if user_input.lower() == 'quit':
                break

            if not user_input:
                continue

            response = generate_response(model, processor, user_input)
            print(f"AI: {response}")

        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
