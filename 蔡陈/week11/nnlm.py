
# nnlm.py 
"""
基于 BERT 的自回归语言模型（AutoRegressive LM）
实现 SFT（Suffix-Prefix Tuning）风格的注意力掩码：
- prefix 位置可以 attend 所有位置（包括 suffix）
- suffix 位置仅能 attend prefix 以及自身（不能 attend 其他 suffix 位置）
训练时：
- 支持两种语料格式：
    1) 每行：prefix<TAB>suffix  （优先采用）
    2) 每行为完整文本（如小说段落）——脚本会自动按 token 级别把行拆分为 prefix/suffix：
         prefix = 前 synthetic_prefix_ratio 比例的 token（可配置），suffix = 剩余 token
"""

import os
os.environ["USE_TORCH"] = "ON"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path
import math
import random
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertConfig, logging

logging.set_verbosity_error()  # 减少 transformers 的日志量


# -------------------------
# Utils / Config
# -------------------------
def default_config():
    return {
        "bert_path": "bert-base-chinese",   # 当本地有 vocab.txt 时会用本地 tokenizer
        "max_length": 256,
        "batch_size": 8,
        "epochs": 10,
        "lr": 2e-5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vocab_file": "vocab.txt",   # if exists, use it
        "corpus": "corpus_1.txt",
        "save_dir": "./checkpoints",
        "seed": 42,
        "grad_accum_steps": 1,
        "log_every": 10,
        "max_generate_len": 64,
        "pad_to_max_length": True,
        # 新增：当行没有 tab 时，按比例切分 prefix / suffix
        "synthetic_prefix_ratio": 0.30  # prefix 占 token 数的比例（0~1），可通过 config 覆盖
    }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset
# -------------------------
class SFTCorpusDataset(Dataset):
    """
    支持：
      - 行格式为: prefix \\t suffix  （优先）
      - 行为完整文本（如小说段落）：按 token 切分为 prefix / suffix（synthetic_prefix_ratio）
    返回：
      dict: input_ids, attention_mask, token_type_ids, prefix_len, raw_len
    """
    def __init__(self, fname, tokenizer, max_length=256, synthetic_prefix_ratio=0.3):
        self.lines = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic_prefix_ratio = float(synthetic_prefix_ratio)

        p = Path(fname)
        if not p.exists():
            raise FileNotFoundError(f"Corpus file not found: {fname}")

        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if ln is None:
                    continue
                # Skip empty lines
                if ln.strip() == "":
                    continue
                self.lines.append(ln)

    def __len__(self):
        return len(self.lines)

    def _make_prefix_suffix(self, line):
        """
        1) 有 \t：按 prefix<TAB>suffix
        2) 无 \t：按字符比例切分（避免 vocab 不完整导致 tokenizer id 为 None，从而 decode 崩溃）
        """
        if "\t" in line:
            prefix, suffix = line.split("\t", 1)
            return prefix, suffix

        # 无 tab：按字符切分
        s = line.strip("\n")
        if not s:
            return "", ""

        # 太短就全部当 suffix
        if len(s) <= 2:
            return "", s

        ratio = float(self.synthetic_prefix_ratio)
        ratio = max(0.0, min(1.0, ratio))

        cut = int(len(s) * ratio)
        # 保证至少切出 1 个字符，且 suffix 至少 1 个字符
        cut = max(0, min(len(s) - 1, cut))

        if cut == 0:
            return "", s

        return s[:cut], s[cut:]

    def __getitem__(self, idx):
        line = self.lines[idx]
        prefix_text, suffix_text = self._make_prefix_suffix(line)

        # Build full text: prefix + suffix; we rely on tokenizer to add special tokens
        full_text = prefix_text + suffix_text  # if prefix_text == "" this is suffix_text

        # compute encodings
        # prefix_enc without special tokens to count prefix token length
        if prefix_text != "":
            prefix_enc = self.tokenizer(prefix_text, add_special_tokens=False)
            ids = prefix_enc.get("input_ids", [])
            # 兜底：过滤 None
            ds = [i for i in ids if i is not None]
            prefix_token_count = len(ids)
        else:
            prefix_token_count = 0

        full_enc = self.tokenizer(full_text, max_length=self.max_length,
                          truncation=True, add_special_tokens=True)

        input_ids = full_enc["input_ids"]
        input_ids = [i if i is not None else 0 for i in input_ids]  # None -> 0 (pad)
        token_type_ids = full_enc.get("token_type_ids", [0] * len(input_ids))
        attention_mask = full_enc.get("attention_mask", [1] * len(input_ids))
        raw_len = len(input_ids)

        # Edge case: if tokenizer adds special tokens at front, prefix_token_count aligns to token positions after special tokens?
        # We defined prefix_token_count as token count in prefix_text without special tokens.
        # When tokenizer adds special tokens (e.g., [CLS]) at front, prefix positions in final input shift by +1.
        # To keep consistent with build_sft_extended_mask which uses prefix_len measured w.r.t. token positions excluding special tokens,
        # we maintain prefix_len as prefix_token_count (i.e., number of non-special tokens from prefix).
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "prefix_len": prefix_token_count,
            "raw_len": raw_len
        }


def collate_fn(batch, pad_id=0, max_len=None, pad_to_max_length=True):
    """
    batch: list of dicts from __getitem__
    输出：padded tensors + prefix_lens list
    """
    bsz = len(batch)
    if max_len is None:
        max_len = max(x["raw_len"] for x in batch)
    if pad_to_max_length:
        max_len = max_len

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    token_type_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    prefix_lens = []

    for i, ex in enumerate(batch):
        l = ex["raw_len"]
        input_ids[i, :l] = ex["input_ids"]
        attention_mask[i, :l] = ex["attention_mask"]
        token_type_ids[i, :l] = ex["token_type_ids"]
        prefix_lens.append(int(ex["prefix_len"]))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "prefix_lens": torch.tensor(prefix_lens, dtype=torch.long)
    }


# -------------------------
# SFT 掩码构建函数
# -------------------------
def build_sft_extended_mask(batch_attention_mask, batch_prefix_lens, device=None):
    """
    构建扩展 attention mask，形状 (batch, 1, seq_len, seq_len)
    掩码值采用 -10000.0（大负）来屏蔽，0 表示不屏蔽。
    规则（对每个序列 i）：
        - query pos q in prefix (q < prefix_len): allow all j in [0, real_len)
        - query pos q in suffix (q >= prefix_len and q < real_len): allow j in [0, prefix_len) and j == q
        padding positions remain masked
    """
    batch, seq_len = batch_attention_mask.shape
    if device is None:
        device = batch_attention_mask.device
    NEG = -10000.0
    extended = torch.full((batch, 1, seq_len, seq_len), NEG, dtype=torch.float, device=device)

    for b in range(batch):
        real_len = int(batch_attention_mask[b].sum().item())
        prefix_len = int(batch_prefix_lens[b].item())
        if prefix_len > real_len:
            prefix_len = real_len

        for q in range(real_len):
            if q < prefix_len:
                extended[b, 0, q, :real_len] = 0.0
            else:
                if prefix_len > 0:
                    extended[b, 0, q, :prefix_len] = 0.0
                extended[b, 0, q, q] = 0.0
        # padding positions remain NEG
    return extended


# -------------------------
# Model
# -------------------------
class BertALM(nn.Module):
    def __init__(self, bert_path_or_config, tokenizer, device="cpu"):
        super(BertALM, self).__init__()
        if isinstance(bert_path_or_config, (str, Path)):
            self.bert = BertModel.from_pretrained(str(bert_path_or_config))
            self.config = self.bert.config
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
            self.config = bert_path_or_config
        else:
            raise ValueError("bert_path_or_config must be a path or BertConfig")

        self.hidden_size = self.config.hidden_size
        self.vocab_size = tokenizer.vocab_size
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        try:
            self.lm_head.weight = self.bert.embeddings.word_embeddings.weight
        except Exception:
            pass

        self.device = device
        self.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids=None, prefix_lens=None):
        embedding_output = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attn = build_sft_extended_mask(attention_mask, prefix_lens, device=input_ids.device)
        encoder_outputs = self.bert.encoder(embedding_output,
                                            attention_mask=extended_attn,
                                            head_mask=[None] * self.config.num_hidden_layers,
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            return_dict=True)
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits


# -------------------------
# Loss helper：只计算 suffix 部分
# -------------------------
def compute_loss_and_metrics(logits, input_ids, prefix_lens, attention_mask, ignore_index=-100):
    bsz, seq_len, vocab = logits.shape
    device = logits.device
    labels = torch.full((bsz, seq_len), ignore_index, dtype=torch.long, device=device)

    for b in range(bsz):
        real_len = int(attention_mask[b].sum().item())
        p = int(prefix_lens[b].item())
        if p < real_len:
            labels[b, p:real_len] = input_ids[b, p:real_len]

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    loss = loss_fct(logits.view(-1, vocab), labels.view(-1))
    n_valid = (labels != ignore_index).sum().item()
    if n_valid > 0:
        loss_val = loss / n_valid
    else:
        loss_val = loss * 0.0
    return loss_val, n_valid


# -------------------------
# Training & Generation
# -------------------------
def train_epoch(model, dataloader, optimizer, scheduler=None, config=None, epoch=0):
    model.train()
    device = config["device"]
    total_loss = 0.0
    total_tokens = 0
    log_every = config.get("log_every", 10)
    grad_accum_steps = config.get("grad_accum_steps", 1)

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        prefix_lens = batch["prefix_lens"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids, prefix_lens=prefix_lens)
        loss, n_valid = compute_loss_and_metrics(logits, input_ids, prefix_lens, attention_mask)
        loss = loss / float(grad_accum_steps)
        loss.backward()
        if step % grad_accum_steps == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += float(loss.item() * grad_accum_steps) * n_valid
        total_tokens += n_valid

        if step % log_every == 0:
            avg = (total_loss / total_tokens) if total_tokens > 0 else 0.0
            print(f"[Epoch {epoch}] step {step}/{len(dataloader)} | avg token loss: {avg:.4f} | tokens so far: {total_tokens}")

    avg_loss = (total_loss / total_tokens) if total_tokens > 0 else 0.0
    return avg_loss


# @torch.no_grad()
# def generate_and_save(model, tokenizer, prompt_prefix, config, max_len=None):
#     model.eval()
#     device = config["device"]
#     max_gen = max_len or config.get("max_generate_len", 64)

#     prefix_enc = tokenizer(prompt_prefix, add_special_tokens=False)
#     prefix_ids = prefix_enc["input_ids"]

#     if tokenizer.cls_token is not None:
#         seq = [tokenizer.cls_token_id] + prefix_ids
#     else:
#         seq = prefix_ids.copy()

#     if tokenizer.sep_token_id is not None:
#         seq = seq + [tokenizer.sep_token_id]

#     for step in range(max_gen):
#         if len(seq) > config["max_length"]:
#             break
#         input_ids = torch.tensor([seq], dtype=torch.long, device=device)
#         attention_mask = torch.ones_like(input_ids, device=device)
#         token_type_ids = torch.zeros_like(input_ids, device=device)
#         prefix_len = torch.tensor([len(prefix_ids)], dtype=torch.long, device=device)

#         logits = model(input_ids, attention_mask, token_type_ids, prefix_len)
#         last_logit = logits[0, -1, :]
#         next_id = int(torch.argmax(last_logit).item())
#         seq.append(next_id)
#         if tokenizer.sep_token_id is not None and next_id == tokenizer.sep_token_id:
#             break

#     out_ids = seq.copy()
#     if tokenizer.cls_token_id is not None and out_ids and out_ids[0] == tokenizer.cls_token_id:
#         out_ids = out_ids[1:]
#     generated_text = tokenizer.decode(out_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
#     return generated_text

import torch.nn.functional as F

@torch.no_grad()
def generate_and_save(model, tokenizer, prompt_prefix, config, max_len=None,
                      temperature=1.0, top_k=50, top_p=0.9,
                      repetition_penalty=1.15, no_repeat_ngram_size=3,
                      max_repeat_same_token=5):
    model.eval()
    device = config["device"]
    max_gen = max_len or config.get("max_generate_len", 64)

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]

    # seq: [CLS] + prefix
    seq = []
    if tokenizer.cls_token_id is not None:
        seq.append(tokenizer.cls_token_id)
    seq.extend(prefix_ids)

    def build_causal_extended_mask(attn_1d):
        L = attn_1d.size(1)
        tril = torch.tril(torch.ones((L, L), device=attn_1d.device))
        pad = (1.0 - attn_1d.float()).view(1, 1, 1, L)
        causal = tril.view(1, 1, L, L)
        causal = causal * (1.0 - pad)
        neg = -10000.0
        return (1.0 - causal) * neg

    def apply_repetition_penalty(logits, generated_ids, penalty):
        # logits: (vocab,)
        if penalty is None or penalty <= 1.0:
            return logits
        unique_ids = set(generated_ids)
        for tid in unique_ids:
            if logits[tid] < 0:
                logits[tid] *= penalty
            else:
                logits[tid] /= penalty
        return logits

    def ban_no_repeat_ngram(logits, generated_ids, n):
        if n is None or n <= 0:
            return logits
        if len(generated_ids) < n:
            return logits
        # 当前 (n-1)-gram 前缀
        prefix = tuple(generated_ids[-(n-1):])
        # 收集所有出现过的 n-gram
        banned = set()
        for i in range(len(generated_ids) - n + 1):
            ng = tuple(generated_ids[i:i+n])
            if ng[:-1] == prefix:
                banned.add(ng[-1])
        for tid in banned:
            logits[tid] = -float("inf")
        return logits

    def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
        # logits: (vocab,)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < kth, torch.tensor(-float("inf"), device=logits.device), logits)

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)

            # 移除累积概率超过 top_p 的部分
            cutoff = cum > top_p
            # 保留第一个超过 top_p 的 token
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_logits = torch.where(cutoff, torch.tensor(-float("inf"), device=logits.device), sorted_logits)
            logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
        return logits

    # 用于检测“同 token 连续重复”
    last_token = None
    repeat_count = 0

    for _ in range(max_gen):
        if len(seq) >= config["max_length"]:
            break

        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        token_type_ids = torch.zeros_like(input_ids, device=device)

        embedding_output = model.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        extended_attn = build_causal_extended_mask(attention_mask)
        enc = model.bert.encoder(
            embedding_output,
            attention_mask=extended_attn,
            head_mask=[None] * model.config.num_hidden_layers,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        hidden = enc.last_hidden_state
        logits = model.lm_head(hidden)[0, -1, :]  # (vocab,)

        # 1) 温度
        if temperature is not None and temperature > 0:
            logits = logits / temperature

        # 2) 重复惩罚（针对已经生成的部分，不含 CLS）
        generated_part = seq[1:] if (tokenizer.cls_token_id is not None and len(seq) > 0 and seq[0] == tokenizer.cls_token_id) else seq
        logits = apply_repetition_penalty(logits, generated_part, repetition_penalty)

        # 3) no-repeat ngram
        logits = ban_no_repeat_ngram(logits, generated_part, no_repeat_ngram_size)

        # 4) top-k/top-p 过滤
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)

        # 如果分布全是 nan 或者全 0，直接 break
        if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
            break

        next_id = int(torch.multinomial(probs, num_samples=1).item())

        # 连续重复保护：如果同一个 token 连续太多次，强制“降低它概率”再采样一次
        if last_token is not None and next_id == last_token:
            repeat_count += 1
        else:
            repeat_count = 0
            last_token = next_id

        if repeat_count >= max_repeat_same_token:
            # 强行禁止该 token 再采一次
            probs2 = probs.clone()
            probs2[next_id] = 0.0
            if float(probs2.sum().item()) > 0:
                probs2 = probs2 / probs2.sum()
                next_id = int(torch.multinomial(probs2, num_samples=1).item())
            repeat_count = 0
            last_token = next_id

        seq.append(next_id)

        if tokenizer.sep_token_id is not None and next_id == tokenizer.sep_token_id:
            break

    out_ids = seq
    if tokenizer.cls_token_id is not None and out_ids and out_ids[0] == tokenizer.cls_token_id:
        out_ids = out_ids[1:]
    text = tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return text.replace(" ", "")

# -------------------------
# Build dataset, tokenizer, model
# -------------------------
# def build_tokenizer_and_model(config):
#     if Path(config["vocab_file"]).exists():
#         tokenizer = BertTokenizer(vocab_file=str(config["vocab_file"]), do_lower_case=False)
#     else:
#         tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
#     model = BertALM(config["bert_path"], tokenizer, device=config["device"])
#     return model, tokenizer

# vocab.txt 中的BERT WordPiece 词表（缺少 [UNK] / 特殊符号不全 / 存在空行 / 编码异常），导致 tokenizer 在转换字符为 id 时出现 None。
def build_tokenizer_and_model(config):
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    model = BertALM(config["bert_path"], tokenizer, device=config["device"])
    return model, tokenizer


def build_dataset_and_loader(tokenizer, config):
    ds = SFTCorpusDataset(config["corpus"], tokenizer, max_length=config["max_length"],
                          synthetic_prefix_ratio=config.get("synthetic_prefix_ratio", 0.3))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=True,
                    collate_fn=lambda b: collate_fn(b, pad_id=pad_id, max_len=config["max_length"],
                                                    pad_to_max_length=config["pad_to_max_length"]))
    return ds, dl


# -------------------------
# Main training routine
# -------------------------
def train_and_eval(config):
    set_seed(config["seed"])
    device = config["device"]
    os.makedirs(config["save_dir"], exist_ok=True)

    model, tokenizer = build_tokenizer_and_model(config)
    dataset, dataloader = build_dataset_and_loader(tokenizer, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    print("Start training:", time.asctime())
    for epoch in range(1, config["epochs"] + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler=None, config=config, epoch=epoch)
        print(f"Epoch {epoch} finished, avg token loss = {avg_loss:.6f}")
        ckpt_path = Path(config["save_dir"]) / f"nnlm_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    prompts = ["北京的夜空", "在图书馆里", "未来科技的趋势"]
    for p in prompts:
        out = generate_and_save(model, tokenizer, p, config)
        print("Prompt:", p)
        print("Generated:", out)
        print("-----")

    return model, tokenizer


# -------------------------
# Entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="json config file (optional)")
    args = parser.parse_args()

    cfg = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)

    print("Config:", json.dumps(cfg, indent=2, ensure_ascii=False))
    model, tokenizer = train_and_eval(cfg)


if __name__ == "__main__":
    main()
