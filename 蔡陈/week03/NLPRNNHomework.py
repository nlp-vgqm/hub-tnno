#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的RNN网络编写
实现一个网络完成一个简单nlp任务
判断特定字符在文本中的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=64, num_layers=1):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)  # RNN层
        self.classify = nn.Linear(hidden_size, 1)  # 线性层，每个时间步输出一个概率值
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.binary_cross_entropy  # 使用二元交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        
        # 对每个时间步的输出进行分类
        y_pred = torch.zeros(x.size(0), x.size(1))  # 初始化预测结果
        for i in range(x.size(1)):  # 遍历每个时间步
            time_step_out = self.classify(rnn_out[:, i, :])  # (batch_size, hidden_size) -> (batch_size, 1)
            y_pred[:, i] = self.activation(time_step_out).squeeze()  # 应用激活函数
        
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

# 修改字符集，包含更多中文字符和英文字符
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # 英文字母和数字
    chinese_chars = "你我他她它的是在了有和就这那要会以可对能而于之"  # 常用中文字符
    chars += chinese_chars
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 标签是一个长度为sentence_length的向量，标记特定字符出现的位置
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    
    # 创建位置标签向量，长度为sentence_length
    y = [0] * sentence_length
    
    # 定义要检测的特定字符
    target_chars = ["你", "我", "他", "A", "B", "1", "2"]
    
    # 标记特定字符出现的位置
    for i, char in enumerate(x):
        if char in target_chars:
            y[i] = 1  # 在该位置标记为1
    
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y

# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    
    total_positions = y.numel()  # 总位置数
    positive_positions = y.sum().item()  # 正样本位置数
    print(f"本次预测集中共有{int(positive_positions)}个正样本位置，{int(total_positions - positive_positions)}个负样本位置")
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 将预测值转换为二进制预测
        y_pred_binary = (y_pred > 0.5).float()
        
        # 计算准确率
        correct = (y_pred_binary == y).sum().item()
        wrong = total_positions - correct
    
    accuracy = correct / (correct + wrong)
    print(f"正确预测位置数：{correct}, 总位置数：{total_positions}, 正确率：{accuracy:.4f}")
    return accuracy

def main():
    # 配置参数
    epoch_num = 20  # 增加训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 50  # 每个字的维度
    sentence_length = 10  # 增加样本文本长度
    learning_rate = 0.001  # 学习率
    
    # 建立字表
    vocab = build_vocab()
    print(f"字符表大小: {len(vocab)}")
    
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    print("模型结构:")
    print(model)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    writer = open("rnn_vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 50  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    
    # 处理输入字符串，确保长度为sentence_length
    processed_strings = []
    for input_string in input_strings:
        if len(input_string) > sentence_length:
            # 截断过长的字符串
            processed_string = input_string[:sentence_length]
        elif len(input_string) < sentence_length:
            # 用空格填充过短的字符串
            processed_string = input_string + ' ' * (sentence_length - len(input_string))
        else:
            processed_string = input_string
        processed_strings.append(processed_string)
    
    x = []
    for input_string in processed_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    
    target_chars = ["你", "我", "他", "A", "B", "1", "2"]
    
    for i, input_string in enumerate(processed_strings):
        predictions = result[i]
        print(f"\n输入：{input_string}")
        print("字符位置: ", " ".join([f"{j:2d}" for j in range(len(input_string))]))
        print("字符内容: ", " ".join([f" {char}" for char in input_string]))
        print("预测概率: ", " ".join([f"{prob:.2f}" for prob in predictions]))
        print("预测位置: ", " ".join([" 1 " if prob > 0.5 else " 0 " for prob in predictions]))
        
        # 显示实际的目标字符位置
        actual_positions = []
        for j, char in enumerate(input_string):
            if char in target_chars:
                actual_positions.append(j)
        print(f"实际目标字符位置: {actual_positions}")

if __name__ == "__main__":
    main()
    
    # 修改测试集，包含更多样化的例子
    test_strings = [
        "ABC123你我他",
        "helloAworld", 
        "测试B字符串",
        "这是1个例子",
        "没有目标字符",
        "你好好好你你",
        "12ABABAB12",
        "随机文本生成"
    ]
    
    # 确保测试字符串长度合适
    test_strings = [s.ljust(10, ' ')[:10] for s in test_strings]
    
    predict("rnn_model.pth", "rnn_vocab.json", test_strings)