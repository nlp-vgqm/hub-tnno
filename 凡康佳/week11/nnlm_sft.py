#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer, T5Model
import json
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.encoder=BertModel.from_pretrained(r"E:\python\学习相关\第六周 语言模型\week6 语言模型和预训练\bert-base-chinese",return_dict=False,attn_implementation='eager')
        self.classify = nn.Linear(768, len(vocab))
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,prompt_len, x, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        if y is not None:
            mask=build_mask(prompt_len,x.shape[0],x.shape[1])
            # mask=mask.cuda()
            x,_=self.encoder(x,attention_mask=mask)
        else:
            x,_=self.encoder(x)
        # print(x.shape)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        # print(y_pred.shape)
        # print(y.shape)
        if y is not None:
            # print(y_pred.view(-1, y_pred.shape[-1]).shape)
            # print(y.view(-1).shape)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index
    return vocab

def build_mask(prompt_lists,batch_size,window_size):
    masks=[]
    for prompt in prompt_lists:
        mask1=torch.ones(window_size,prompt)

        mask2=torch.zeros(prompt,window_size-prompt)
        mask4=torch.tril(torch.ones(window_size-prompt,window_size-prompt))
        mask=torch.cat([mask1,torch.cat([mask2,mask4],dim=0)],dim=1)
        masks.append(mask)

    mask=torch.stack(masks)
    if torch.cuda.is_available():
        mask=mask.cuda()
    return mask

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            json_data = json.loads(line)
            title=json_data['title']
            content=json_data['content']
            corpus.append([title,content]) # 多一个错位的时候好凑一个整数
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, corpus,window_size):

    prompt=tokenizer.encode(corpus[0],add_special_tokens=False)

    answer=tokenizer.encode(corpus[1][:window_size],add_special_tokens=False)
    answer=answer[:window_size-len(prompt)]
    if len(prompt)+len(answer)<window_size:
        answer=answer+[tokenizer.pad_token_id]*(window_size-len(answer)-len(prompt))
    

    x=[tokenizer.cls_token_id]+prompt+[tokenizer.sep_token_id]+answer+[tokenizer.sep_token_id]
    y=[-1]*len(prompt)+[-1]+answer+[tokenizer.sep_token_id]+[-1]
    
    return len(prompt)+1,x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch_size, tokenizer, corpus,window_size):
    dataset_x = []
    dataset_y = []
    prompt_lens=[]
    for i in range(batch_size):
        choise = random.randint(0, len(corpus) - 1)
        prompt_len,x, y = build_sample(tokenizer, corpus[choise],window_size)
        dataset_x.append(x)
        dataset_y.append(y)
        prompt_lens.append(prompt_len)

    return prompt_lens,torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer, max_tokens=50):
    model.eval()
    input_ids = tokenizer.encode(openings, add_special_tokens=True)
    
    with torch.no_grad():
        while len(input_ids) < max_tokens:
            x = torch.LongTensor([input_ids]).cuda() if torch.cuda.is_available() else torch.LongTensor([input_ids])
            logits = model(None, x)[0, -1, :]  # 假设模型返回 (batch, seq, vocab)
            next_id = sampling_strategy(logits)  # 返回 int
            
            if next_id in {tokenizer.pad_token_id, tokenizer.eos_token_id}:
                break
            input_ids.append(next_id)
    
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 640   #每轮训练总共训练的样本总数
    char_dim = 256        #每个字的维度
    window_size = 50       #样本文本长度
    vocab = build_vocab(r"E:\python\学习相关\第六周 语言模型\week6 语言模型和预训练\bert-base-chinese\vocab.txt")       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(vocab, char_dim)    #建立模型
    tokenizer = BertTokenizer.from_pretrained(r"E:\python\学习相关\第六周 语言模型\week6 语言模型和预训练\bert-base-chinese")
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(int(train_sample / batch_size)):
            prompt_len,x, y = build_dataset(batch_size, tokenizer, corpus,window_size) #构建一组训练样本
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()
                optim.zero_grad()    #梯度归零
                loss = model(prompt_len,x, y)   #计算loss
                loss.backward()      #计算梯度
                optim.step()         #更新权重
                watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("朝鲜宣布将于4月发射人造卫星", model, tokenizer))
        print(generate_sentence("冰岛女总理将携夫人访华 为该国首对同性结婚伴侣", model,tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)
