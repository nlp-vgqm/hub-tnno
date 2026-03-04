
import os
os.environ["USE_TORCH"] = "ON"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 屏蔽 TF 的提示信息

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer

class BertALM(nn.Module):
    def __init__(self, config):
        super(BertALM, self).__init__()
        # 加载 BERT 配置，不再使用原有的 Embedding 和 RNN 层
        self.bert = BertModel.from_pretrained(config["bert_path"])
        # 线性层：从 BERT 的隐藏维度映射到整个词表大小
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0) # 0 通常是 [PAD]

    def forward(self, x, y=None):
        # 1. 生成 Causal Mask (下三角掩码)
        # 形状: [batch_size, seq_len, seq_len]
        batch_size, seq_len = x.shape
        # 使用 torch.ones 创建基础矩阵
        # torch.tril 生成下三角矩阵，实现自回归掩码
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device))
        
        # 这里的 mask 维度对齐非常重要
        # BERT 接收 3D mask 时，形状应为 [batch_size, seq_len, seq_len]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        outputs = self.bert(x, attention_mask=mask)
        sequence_output = outputs[0] # [batch_size, seq_len, hidden_size]
        
        # 3. 预测词表概率
        prediction = self.classify(sequence_output) # [batch_size, seq_len, vocab_size]
        
        if y is not None:
            return self.loss(prediction.view(-1, prediction.shape[-1]), y.view(-1))
        return prediction

# 数据处理函数
def build_dataset(corpus_path, tokenizer, max_length):
    dataset = []
    # encoding="utf8"
    with open(corpus_path, encoding="utf8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 编码文本
            ids = tokenizer.encode(line, add_special_tokens=True, max_length=max_length+1, truncation=True, padding='max_length')
            
            # 自回归任务：输入前 n-1 个，预测后 n-1 个
            # x: [CLS, char1, char2, ..., char_n-1]
            # y: [char1, char2, ..., char_n, SEP]
            x = torch.LongTensor(ids[:-1])
            y = torch.LongTensor(ids[1:])
            dataset.append([x, y])
    return dataset

# 模拟配置
config = {
    "bert_path": "bert-base-chinese", # 必须与 vocab.txt 对应
    "max_length": 20,
    "batch_size": 4,
    "lr": 1e-5
}

# --- 以下代码加在 nnlm.py 末尾 ---

def generate_and_save(model, tokenizer, start_text, config, save_path="generated_text.txt"):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(start_text, add_special_tokens=True)[:-1] 
    
    with torch.no_grad():
        for _ in range(config.get("generate_length", 30)):
            x = torch.LongTensor([input_ids]).to(device)
            predictions = model(x) 
            last_token_logits = predictions[0, -1, :]
            
            # --- 修改部分：增加采样随机性，防止复读 ---
            # 设置温度 Temperature，1.0 是原始分布，越小越保守，越大越随机
            temperature = 1.2 
            last_token_logits = last_token_logits / temperature
            
            # 将 logits 转化为概率分布
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # 按照概率分布进行随机采样，而不是死板地选最高分
            predicted_id = torch.multinomial(probs, num_samples=1).item()
            # ---------------------------------------
            
            input_ids.append(predicted_id)
            if predicted_id == tokenizer.sep_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    generated_text = "".join(generated_text.split())
    
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(f"Result: {generated_text}\n")
    return generated_text

# # 1. 修改 train 函数，增加保存逻辑
def train():
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    dataset = build_dataset("corpus_1.txt", tokenizer, config["max_length"])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    model = BertALM(config)
    if torch.cuda.is_available(): model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    model.train()
    for epoch in range(50): # 建议多跑几轮
        for x, y in train_loader:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # --- 关键：训练完一定要保存 ---
    torch.save(model.state_dict(), "bert_lm_model.pth")
    print("模型已保存")
    return model, tokenizer

# 2. 修改主入口
if __name__ == "__main__":
    # 执行训练
    model, tokenizer = train() 
    
    # 执行生成
    test_starts = ["东京", "林默身旁", "未来30年"]
    for start in test_starts:
        # 这里传入刚刚训练好的 model
        generate_and_save(model, tokenizer, start, config)


