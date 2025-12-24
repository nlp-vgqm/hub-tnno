import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import re
import math
from collections import Counter
from torch.utils.data import Dataset, DataLoader


# ==================== 基础Transformer组件 ====================

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.w_o(attn_output)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


# ==================== 改进的数据集 ====================

def create_translation_dataset():
    """创建翻译数据集"""

    # 系统化的翻译对
    translation_pairs = []

    # 1. 基础词汇
    basics = [
        ("你好", "hello"),
        ("谢谢", "thank you"),
        ("再见", "goodbye"),
        ("是的", "yes"),
        ("不是", "no"),
        ("请", "please"),
        ("对不起", "sorry"),
        ("好的", "okay"),
        ("早上好", "good morning"),
        ("晚上好", "good evening"),
        ("晚安", "good night"),
    ]

    # 2. 人称代词 + be动词 + 名词
    pronouns = ["我", "你", "他", "她", "我们", "你们", "他们"]
    be_verbs = ["是", "不是"]
    nouns = ["老师", "学生", "医生", "工程师", "朋友", "中国人", "美国人"]

    for pronoun in pronouns:
        for be_verb in be_verbs:
            for noun in nouns:
                chinese = f"{pronoun}{be_verb}{noun}"
                english = f"{pronoun_to_eng(pronoun)} {be_to_eng(be_verb, pronoun)} {noun_to_eng(noun)}"
                translation_pairs.append({"chinese": chinese, "english": english})

    # 3. 简单动作句
    actions = ["吃", "喝", "看", "读", "写", "学习", "工作", "玩", "喜欢"]
    objects = ["苹果", "水", "书", "信", "英语", "电脑", "游戏", "电影", "音乐"]

    for pronoun in ["我", "你", "他", "她"]:
        for action in actions:
            for obj in objects:
                chinese = f"{pronoun}{action}{obj}"
                english = f"{pronoun_to_eng(pronoun)} {action_to_eng(action)} {obj_to_eng(obj)}"
                translation_pairs.append({"chinese": chinese, "english": english})

    # 4. 疑问句
    questions = [
        ("你叫什么名字？", "what is your name?"),
        ("你来自哪里？", "where are you from?"),
        ("你多大了？", "how old are you?"),
        ("现在几点了？", "what time is it now?"),
        ("这个用英语怎么说？", "how do you say this in english?"),
        ("你会说中文吗？", "can you speak chinese?"),
        ("你喜欢什么颜色？", "what color do you like?"),
        ("今天天气怎么样？", "how is the weather today?"),
        ("你饿了吗？", "are you hungry?"),
        ("你累了吗？", "are you tired?"),
    ]

    translation_pairs.extend([{"chinese": c, "english": e} for c, e in questions])

    # 5. 时间和地点
    times = ["今天", "明天", "昨天", "现在", "早上", "下午", "晚上", "每天"]
    places = ["家", "学校", "公司", "医院", "公园", "超市", "餐厅", "图书馆"]
    actions2 = ["去", "在", "到", "想去"]

    for time in times:
        for action in actions2:
            for place in places:
                chinese = f"{time}{action}{place}"
                english = f"{time_to_eng(time)} {action_to_eng(action)} {place_to_eng(place)}"
                translation_pairs.append({"chinese": chinese, "english": english})

    # 6. 情感和状态
    emotions = [
        ("我爱你", "i love you"),
        ("天气很好", "the weather is good"),
        ("我很高兴", "i am happy"),
        ("我饿了", "i am hungry"),
        ("我累了", "i am tired"),
        ("我想你", "i miss you"),
        ("祝你生日快乐", "happy birthday to you"),
        ("新年快乐", "happy new year"),
        ("我很抱歉", "i am sorry"),
        ("我很兴奋", "i am excited"),
        ("我很无聊", "i am bored"),
        ("我很忙", "i am busy"),
    ]

    translation_pairs.extend([{"chinese": c, "english": e} for c, e in emotions])

    # 7. 数字和颜色
    numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    colors = ["红色", "蓝色", "绿色", "黄色", "黑色", "白色", "紫色", "橙色"]

    for i, num in enumerate(numbers[:5], 1):
        for color in colors:
            chinese = f"{num}个{color}的苹果"
            english = f"{i} {color_to_eng(color)} apple{'s' if i > 1 else ''}"
            translation_pairs.append({"chinese": chinese, "english": english})

    # 8. 更多日常句子
    daily = [
        ("我在家", "i am at home"),
        ("他在学校", "he is at school"),
        ("她在工作", "she is at work"),
        ("我们在学习", "we are studying"),
        ("你们在玩", "you are playing"),
        ("他们在吃饭", "they are eating"),
        ("我想喝水", "i want to drink water"),
        ("我要去超市", "i want to go to the supermarket"),
        ("他喜欢足球", "he likes football"),
        ("她爱音乐", "she loves music"),
    ]

    translation_pairs.extend([{"chinese": c, "english": e} for c, e in daily])

    print(f"创建了 {len(translation_pairs)} 个翻译对")
    return translation_pairs


def pronoun_to_eng(p):
    mapping = {"我": "i", "你": "you", "他": "he", "她": "she",
               "我们": "we", "你们": "you", "他们": "they"}
    return mapping.get(p, p)


def be_to_eng(b, pronoun):
    if b == "是":
        if pronoun == "我":
            return "am"
        elif pronoun in ["你", "我们", "你们", "他们"]:
            return "are"
        elif pronoun in ["他", "她"]:
            return "is"
    else:  # "不是"
        if pronoun == "我":
            return "am not"
        elif pronoun in ["你", "我们", "你们", "他们"]:
            return "are not"
        elif pronoun in ["他", "她"]:
            return "is not"
    return b


def noun_to_eng(n):
    mapping = {"老师": "teacher", "学生": "student", "医生": "doctor",
               "工程师": "engineer", "朋友": "friend", "中国人": "chinese",
               "美国人": "american"}
    return mapping.get(n, n)


def action_to_eng(a):
    mapping = {"吃": "eat", "喝": "drink", "看": "watch", "读": "read",
               "写": "write", "学习": "study", "工作": "work", "玩": "play",
               "喜欢": "like", "去": "go to", "在": "am at", "到": "go to",
               "想去": "want to go to", "爱": "love"}
    return mapping.get(a, a)


def obj_to_eng(o):
    mapping = {"苹果": "apple", "水": "water", "书": "book", "信": "letter",
               "英语": "english", "电脑": "computer", "游戏": "game",
               "电影": "movie", "音乐": "music", "足球": "football"}
    return mapping.get(o, o)


def time_to_eng(t):
    mapping = {"今天": "today", "明天": "tomorrow", "昨天": "yesterday",
               "现在": "now", "早上": "in the morning", "下午": "in the afternoon",
               "晚上": "in the evening", "每天": "every day"}
    return mapping.get(t, t)


def place_to_eng(p):
    mapping = {"家": "home", "学校": "school", "公司": "company",
               "医院": "hospital", "公园": "park", "超市": "supermarket",
               "餐厅": "restaurant", "图书馆": "library"}
    return mapping.get(p, p)


def color_to_eng(c):
    mapping = {"红色": "red", "蓝色": "blue", "绿色": "green",
               "黄色": "yellow", "黑色": "black", "白色": "white",
               "紫色": "purple", "橙色": "orange"}
    return mapping.get(c, c)


class Vocabulary:
    """词汇表"""

    def __init__(self, language='chinese'):
        self.language = language
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.idx = 4

        # 预定义一些常见词
        if language == 'english':
            common_words = [
                'i', 'you', 'he', 'she', 'we', 'they', 'am', 'is', 'are',
                'a', 'an', 'the', 'and', 'but', 'or', 'in', 'on', 'at',
                'to', 'for', 'of', 'with', 'by', 'my', 'your', 'his', 'her',
                'this', 'that', 'what', 'where', 'when', 'why', 'how',
                'can', 'do', 'does', 'have', 'has', 'good', 'bad', 'happy',
                'sad', 'big', 'small', 'red', 'blue', 'green', 'yellow'
            ]
            for word in common_words:
                self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, texts, language='chinese'):
        """构建词汇表"""
        for text in texts:
            if language == 'chinese':
                # 中文按字符分词，但过滤标点
                for char in text:
                    if char not in [' ', '？', '！', '。', '，', '?', '!', '.', ',']:
                        self._add_word(char)
            else:
                # 英文分词
                words = re.findall(r'\b\w+\b', text.lower())
                for word in words:
                    self._add_word(word)

        print(f"{language}词汇表大小: {len(self)}")

    def encode(self, text, max_len=None, language='chinese'):
        """编码文本"""
        if language == 'chinese':
            # 中文：过滤标点，按字符
            text = re.sub(r'[？！。，?！.,]', '', text)
            tokens = ['<sos>'] + [char for char in text if char.strip()] + ['<eos>']
        else:
            # 英文：小写化，分词
            text = text.lower()
            words = re.findall(r'\b\w+\b', text)
            tokens = ['<sos>'] + words + ['<eos>']

        indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]

        if max_len:
            if len(indices) < max_len:
                indices = indices + [self.word2idx['<pad>']] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

        return indices

    def decode(self, indices):
        """解码"""
        tokens = []
        for idx in indices:
            if idx == self.word2idx['<eos>']:
                break
            if idx not in [self.word2idx['<pad>'], self.word2idx['<sos>']]:
                tokens.append(self.idx2word.get(idx, '<unk>'))

        if self.language == 'chinese':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

    def __len__(self):
        return len(self.word2idx)


# ==================== Transformer模型 ====================

class TranslationTransformer(nn.Module):
    """翻译Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256,
                 num_heads=8, num_layers=4, d_ff=1024, dropout=0.1, max_len=100):
        super().__init__()

        self.d_model = d_model

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.positional_encoding = self._create_positional_encoding(d_model, max_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 初始化权重
        self._init_weights()

    def _create_positional_encoding(self, d_model, max_len):
        """创建位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为缓冲区
        self.register_buffer('pe', pe)

        return pe

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 嵌入层特殊初始化
        nn.init.normal_(self.src_embedding.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.normal_(self.tgt_embedding.weight, mean=0, std=self.d_model ** -0.5)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pe[:, :src_emb.size(1), :]
        src_emb = self.dropout(src_emb)

        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        # 解码器
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pe[:, :tgt_emb.size(1), :]
        tgt_emb = self.dropout(tgt_emb)

        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        # 输出
        output = self.output_layer(decoder_output)
        return output


# ==================== 训练和评估函数 ====================

def create_masks(src, tgt, pad_idx=0):
    """创建注意力掩码"""
    # 源序列掩码
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # 目标序列掩码（填充掩码 + 因果掩码）
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)
    tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(0)
    tgt_mask = tgt_pad_mask & ~tgt_causal_mask

    return src_mask, tgt_mask


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        # 创建掩码
        src_mask, tgt_mask = create_masks(src, tgt[:, :-1])

        optimizer.zero_grad()

        # 前向传播
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)

        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)),
                         tgt[:, 1:].reshape(-1))

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            src_mask, tgt_mask = create_masks(src, tgt[:, :-1])

            output = model(src, tgt[:, :-1], src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)),
                             tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=20):
    """贪婪解码翻译"""
    model.eval()

    # 编码源句子
    src_indices = src_vocab.encode(src, max_len, 'chinese')
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

    # 创建源掩码
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)

    # 初始化目标序列
    tgt_indices = [tgt_vocab.word2idx['<sos>']]

    for _ in range(max_len - 1):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)

        # 创建目标掩码
        tgt_mask = (tgt_tensor != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = len(tgt_indices)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
        tgt_mask = tgt_mask & ~causal_mask.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)

        # 贪婪选择下一个词
        next_token = output[0, -1].argmax().item()
        tgt_indices.append(next_token)

        if next_token == tgt_vocab.word2idx['<eos>']:
            break

    # 解码
    translation = tgt_vocab.decode(tgt_indices)
    return translation


def calculate_exact_match(predicted, reference):
    """计算完全匹配率"""
    return predicted.lower() == reference.lower()


def calculate_word_accuracy(predicted, reference):
    """计算词级别准确率"""
    pred_words = predicted.lower().split()
    ref_words = reference.lower().split()

    if not ref_words:
        return 0.0

    correct = 0
    min_len = min(len(pred_words), len(ref_words))

    for i in range(min_len):
        if pred_words[i] == ref_words[i]:
            correct += 1

    return correct / len(ref_words)


# ==================== 主函数 ====================

def main():
    """主函数"""

    print("=" * 80)
    print("中英翻译Transformer验证")
    print("=" * 80)

    # 1. 准备数据
    print("\n1. 准备翻译数据集...")
    translation_pairs = create_translation_dataset()

    print(f"\n数据集大小: {len(translation_pairs)} 个句子对")

    # 显示一些示例
    print("\n示例句子:")
    for i in range(min(3, len(translation_pairs))):
        print(f"  {i + 1}. 中文: {translation_pairs[i]['chinese']}")
        print(f"      英文: {translation_pairs[i]['english']}")

    # 构建词汇表
    src_vocab = Vocabulary('chinese')
    tgt_vocab = Vocabulary('english')

    chinese_texts = [pair['chinese'] for pair in translation_pairs]
    english_texts = [pair['english'] for pair in translation_pairs]

    src_vocab.build_vocab(chinese_texts, 'chinese')
    tgt_vocab.build_vocab(english_texts, 'english')

    print(f"\n中文词汇表大小: {len(src_vocab)}")
    print(f"英文词汇表大小: {len(tgt_vocab)}")

    # 创建数据集类
    class TranslationDataset(Dataset):
        def __init__(self, pairs, src_vocab, tgt_vocab, max_len=20):
            self.pairs = pairs
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            self.max_len = max_len

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            src_indices = self.src_vocab.encode(pair['chinese'], self.max_len, 'chinese')
            tgt_indices = self.tgt_vocab.encode(pair['english'], self.max_len, 'english')

            return {
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long),
                'chinese': pair['chinese'],
                'english': pair['english']
            }

    dataset = TranslationDataset(translation_pairs, src_vocab, tgt_vocab, max_len=20)

    # 分割数据集 (80%训练, 20%验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\n训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")

    # 2. 创建模型
    print("\n2. 创建Transformer模型...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_len=100
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 训练模型
    print("\n3. 开始训练...")

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    num_epochs = 50
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_translation_model.pth')

        # 每10个epoch显示进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")

            # 显示一些示例翻译
            print("\n  示例翻译:")
            model.eval()
            with torch.no_grad():
                # 取几个验证样本
                val_samples = []
                for i in range(min(3, len(val_dataset))):
                    val_samples.append(val_dataset[i])

                for i, sample in enumerate(val_samples):
                    src_sentence = sample['chinese']
                    reference = sample['english']

                    # 生成翻译
                    predicted = greedy_decode(model, src_sentence, src_vocab, tgt_vocab, device)

                    print(f"  中文: {src_sentence}")
                    print(f"  参考: {reference}")
                    print(f"  预测: {predicted}")

                    exact_match = calculate_exact_match(predicted, reference)
                    word_acc = calculate_word_accuracy(predicted, reference)
                    print(f"  完全匹配: {'✓' if exact_match else '✗'}")
                    print(f"  词准确率: {word_acc:.2%}")
                    print("-" * 40)

    # 加载最佳模型
    print("\n加载最佳模型...")
    model.load_state_dict(torch.load('best_translation_model.pth', map_location=device))

    # 4. 最终测试
    print("\n" + "=" * 80)
    print("最终测试结果")
    print("=" * 80)

    model.eval()

    # 在验证集上测试
    print("\n验证集表现:")
    exact_matches = 0
    word_accuracies = []

    with torch.no_grad():
        # 测试所有验证样本
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            src_sentence = sample['chinese']
            reference = sample['english']

            predicted = greedy_decode(model, src_sentence, src_vocab, tgt_vocab, device)

            exact_match = calculate_exact_match(predicted, reference)
            word_acc = calculate_word_accuracy(predicted, reference)

            if exact_match:
                exact_matches += 1
            word_accuracies.append(word_acc)

            # 显示前5个的详细信息
            if i < 5:
                print(f"\n{i + 1}. 中文: {src_sentence}")
                print(f"   参考: {reference}")
                print(f"   预测: {predicted}")
                print(f"   完全匹配: {'✓' if exact_match else '✗'}")
                print(f"   词准确率: {word_acc:.2%}")

    print(f"\n统计结果:")
    print(f"  完全匹配率: {exact_matches / len(val_dataset):.2%}")
    print(f"  平均词准确率: {np.mean(word_accuracies):.2%}")

    # 5. 泛化测试
    print("\n" + "=" * 80)
    print("泛化能力测试")
    print("=" * 80)

    test_cases = [
        ("我爱你", "i love you"),
        ("今天天气很好", "the weather is good today"),
        ("我正在学习英语", "i am studying english"),
        ("她是一个好老师", "she is a good teacher"),
        ("我们明天去公园", "we will go to the park tomorrow"),
        ("他喜欢红色", "he likes red"),
        ("我有两个苹果", "i have two apples"),
        ("猫在桌子上", "the cat is on the table"),
        ("我想喝水", "i want to drink water"),
        ("生日快乐", "happy birthday"),
        ("你是我的朋友", "you are my friend"),
        ("他们在看电视", "they are watching tv"),
        ("这本书很有趣", "this book is interesting"),
        ("我很高兴见到你", "i am happy to meet you"),
        ("明天见", "see you tomorrow"),
    ]

    print("\n新句子翻译测试:")
    correct = 0
    total_word_acc = 0

    for i, (chinese, reference) in enumerate(test_cases):
        predicted = greedy_decode(model, chinese, src_vocab, tgt_vocab, device)

        exact_match = calculate_exact_match(predicted, reference)
        word_acc = calculate_word_accuracy(predicted, reference)

        if exact_match:
            correct += 1
        total_word_acc += word_acc

        print(f"\n{i + 1}. 中文: {chinese}")
        print(f"   参考: {reference}")
        print(f"   预测: {predicted}")
        print(f"   完全匹配: {'✓' if exact_match else '✗'}")
        print(f"   词准确率: {word_acc:.2%}")

    print(f"\n泛化测试结果:")
    print(f"  完全匹配率: {correct / len(test_cases):.2%}")
    print(f"  平均词准确率: {total_word_acc / len(test_cases):.2%}")

    # 6. 模型分析
    print("\n" + "=" * 80)
    print("模型分析")
    print("=" * 80)

    print("\n模型架构:")
    print(f"  编码器层数: {len(model.encoder_layers)}")
    print(f"  解码器层数: {len(model.decoder_layers)}")
    print(f"  注意力头数: {model.encoder_layers[0].self_attn.num_heads}")
    print(f"  模型维度: {model.d_model}")
    print(f"  前馈网络维度: {model.encoder_layers[0].feed_forward.linear1.out_features}")

    # 测试一个简单例子
    print("\n注意力机制测试:")
    test_src = "我爱你"
    test_tgt = "i love you"

    print(f"  测试句子: {test_src} -> {test_tgt}")

    # 编码
    src_indices = src_vocab.encode(test_src, 10, 'chinese')
    tgt_indices = tgt_vocab.encode(test_tgt, 10, 'english')

    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    tgt_tensor = torch.tensor(tgt_indices[:-1], dtype=torch.long).unsqueeze(0).to(device)

    src_mask, tgt_mask = create_masks(src_tensor, tgt_tensor)

    with torch.no_grad():
        output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)

    print(f"  输入形状: src={src_tensor.shape}, tgt={tgt_tensor.shape}")
    print(f"  输出形状: {output.shape}")

    # 检查位置编码
    print("\n位置编码验证:")
    test_input = torch.zeros(1, 5, model.d_model).to(device)
    pe_output = test_input + model.pe[:, :5, :]

    print(f"  位置编码形状: {pe_output.shape}")
    print(f"  位置0（前3维）: {pe_output[0, 0, :3].cpu().detach().numpy().round(3)}")
    print(f"  位置1（前3维）: {pe_output[0, 1, :3].cpu().detach().numpy().round(3)}")
    print(f"  位置4（前3维）: {pe_output[0, 4, :3].cpu().detach().numpy().round(3)}")

    print("\n" + "=" * 80)
    print("翻译Transformer验证完成！")
    print("=" * 80)


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()