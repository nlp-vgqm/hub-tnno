import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SearchChar(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim, char_len):
        super(SearchChar, self).__init__()
        # self.char_embedding_dim = char_embedding_dim
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim,padding_idx=0)
        self.rnn = nn.RNN(input_size=char_embedding_dim, hidden_size=char_embedding_dim,bias=False,batch_first=True)
        self.line = nn.Linear(char_embedding_dim, 1)
        self.activation= torch.sigmoid
        self.loss= F.mse_loss

        
    def forward(self, char_inputs,target=None):
        # print(char_inputs.shape)
        char_embedding=self.char_embedding(char_inputs)
        # print(char_embedding.shape)
        output,_=self.rnn(char_embedding)
        # print(output.shape)
        output=self.line(output)
        # print(output.shape)
        output=self.activation(output)
        # print(output.shape)
        output=output.squeeze(2)
        # print(output.shape)
        if target is not None:
            return self.loss(output,target)
        else:
            return output


    
vocab_dict={
    chr(i):i-96 for i in range(97,123)
}
vocab_dict['[UNK]']=27
vocab_dict['[PAD]']=0

# print(vocab_dict)

def get_vocab_value(string,dim):
    target=[]
    for i in string:
        if i not in vocab_dict:
            target.append(vocab_dict['[UNK]'])
            continue
        target.append(vocab_dict[i])
    if len(target)<dim:
        target.extend([vocab_dict['[PAD]']]*(dim-len(target)))
    return target



lr=0.001
batch=1000
char_len=10
em_dim=10
# 随机生成一些包含a的字符串
def generate_data(batch_size, seq_len):
    targets = []
    strings = []
    for _ in range(batch_size):
        # 50%概率生成含'a'的字符串
        if np.random.rand() > 0.5:
            # 确保至少包含一个'a'
            chars = [chr(np.random.randint(97, 123)) for _ in range(seq_len-1)]
            insert_pos = np.random.randint(0, seq_len)
            chars.insert(insert_pos, 'a')
            string = ''.join(chars[:seq_len])
        else:
            # 生成不含'a'的字符串
            string = ''.join([chr(np.random.randint(98, 123)) for _ in range(seq_len)])
        targets.append([int(i=='a') for i in string])
        strings.append(string)
    
    return (
        torch.FloatTensor(targets),
        strings,
        torch.LongTensor([get_vocab_value(s, char_len) for s in strings])
    )

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,strings,y = generate_data(test_sample_num, char_len)
    with torch.no_grad():
        y_pred = model(y)

    correct_num=0

    for i,j in zip(x,y_pred):
        same=True
        for m,n in zip(i,j):
            if (m>0.5 and n>0.5) or (m<0.5 and n<0.5):
                continue
            else:
                same=False
                break
        if same:
            correct_num+=1
        
    return test_sample_num,correct_num/test_sample_num



model=SearchChar(28,em_dim,char_len)
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
model.train()
watch=[]
acc=[]
log=[]
for i in tqdm(range(1000)):
    batch_list=generate_data(batch,char_len)
    loss=model.forward(batch_list[2],batch_list[0])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    test_sample_num,test_acc_rate=evaluate(model)
    log.append(f'第『{i+1}』轮测试，总数{test_sample_num}准确率{test_acc_rate}')
    acc.append(test_acc_rate)
    watch.append(loss.item())
    # if all(acc[-10:-1]>=1.0):
    #     print('连续10次准确率都达到1.0，提前结束训练')

# for i in log:
#     print(i)

# 实际输出检查一下
model.eval()
test_sample_num = 10
x,strings,y = generate_data(test_sample_num, char_len)
with torch.no_grad():
    y_pred = model(y)

correct_num=0

for i in range(test_sample_num):
    print(f'字符串{strings[i]}\n预测结果：{y_pred[i]}\n等效验结果：{[int(i>0.5) for i in y_pred[i]]}')





plt.plot(range(len(acc)), [l for l in acc], label="acc")  # 画acc曲线
plt.plot(range(len(watch)), [l for l in watch], label="loss")  # 画loss曲线
plt.legend()
plt.show()
