"""
    模型训练主流程
"""
import json
import pandas as pd
import torch

from NLPtest.第七周作业.evaluate import *
from loader import *
import numpy as np



def main(models, pooling):
    #建立rnn模型需要的字表
    train_vocab, test_vocab = datacsv(config['data_path'], config['test_size'], config['random_state'],
                                      config['shuffle'])
    texts = []
    for i in train_vocab.keys():
        texts.append(i)
    vocab = text_to_vocab(texts)
    #建立rnn模型
    if models == 'rnn':
        model = model_rnn(config['char_dim'], text_long(texts), len(vocab), pooling)
    elif models == 'cnn':
        model = model_cnn(config['char_dim'], text_long(texts), len(vocab), pooling)
    elif models == 'lstm':
        model = model_lstm(config['char_dim'], text_long(texts), len(vocab), pooling)
    #创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=config['learning_num'])
    log = []
    #rnn训练过程
    for epoch in range(config['epoch_num']):
        model.train()
        watch_loss = []     #记录训练过程中每个批次的损失值
        for j in range(int(config['train_sample']/config['batch_size'])):
            x, y = datatrain_rnn(train_vocab, text_to_vocab(texts), text_long(texts))
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = eval_model(model, config['data_path'], config['test_size'], config['random_state'], config['shuffle'], texts)
        log.append([models, acc, np.mean(watch_loss), config['learning_num'], config['char_dim'], config["batch_size"], pooling])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return log

if __name__ == '__main__':
    best_of_each_model = []
    for model in ["rnn", 'cnn', 'lstm']:
        logs = []
        models = model
        for lr in [1e-3, 1e-4]:
            config['learning_num'] = lr
            for char_dim in [128]:
                config['char_dim'] = char_dim
                for batch_size in [64, 128]:
                    config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        pooling = pooling_style
                        log = main(models, pooling)
                        log = max(log, key=lambda x: x[1])
                        logs.append(log)
        result = max(logs, key=lambda x: x[1])
        best_of_each_model.append({
            'model': model,
            'acc': result[1],
            'learning_num': result[3],
            'char_dim': result[4],
            'batch_size': result[5],
            'pooling_style': result[6]
        })
    df = pd.DataFrame(best_of_each_model)
    df = df.sort_values(by=['acc'], ascending=False)
    df.to_excel("vocab.xlsx")

    print('每个模型的最佳结果')
    print(df)
