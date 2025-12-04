"""
使用 lstm 进行文本分类操作
使用 bert 的 embedding 和 vocab 词汇表
"""
import torch
from torch.utils import data
from datasets import load_dataset
from torch import nn
from transformers import BertTokenizer, BertModel

model_path = "E:\\study\\AI\\nlp_data\\model\\bert-base-chinese"
tokenize = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=model_path)

class LstmDataset(data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        review = self.dataset[idx]['review']
        label = self.dataset[idx]['label']

        return review, label


class LstmClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = bert_model.embeddings.word_embeddings
        self.net = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, hidden, cell):
        embed_X = self.embedding(X) # [16, 50] --> [16, 50, 768]
        rnn_X, (hidden, cell) = self.net(embed_X, (hidden, cell))

        output = self.fc(rnn_X[:, -1]) # 最后一个时间步特征来计算 [16, 50, 768] -> [16, 2]
        return output, (hidden, cell)


    def init_hidden_cell(self, batchSize):
        hidden = torch.zeros(size=(self.num_layers, batchSize, self.hidden_size))
        cell = torch.zeros(size=(self.num_layers, batchSize, self.hidden_size))

        return hidden, cell


def lstm_collate_fn(records):
    reviews = [record[0] for record in records]
    labels = [record[1] for record in records]

    # 文本张量化
    encode_input = tokenize.batch_encode_plus(batch_text_or_text_pairs=reviews, max_length=max_len, padding=True, truncation=True, return_tensors='pt', return_length=True)

    input_ids = encode_input['input_ids']
    labels = torch.LongTensor(labels)

    return input_ids, labels

max_len = 50
epochs, batch_size, lr = 3, 16, 1e-3
def train_lstm():
    # 加载数据
    train_dataset = LstmDataset()

    # 定义模型和相关函数
    net = LstmClassifier(input_size=768, hidden_size=768, output_size=2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    for epoch in range(epochs):
        data_iter = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lstm_collate_fn, drop_last=True)
        for i, (input_ids, labels) in enumerate(data_iter):
            hidden, cell = net.init_hidden_cell(batch_size)
            y_hat, (hidden, cell) = net(input_ids, hidden, cell)
            l = loss(y_hat, labels)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if i % 100 == 0: # NOTE: 这里不是统计所有, 是统计每个100次小迭代之后的准确率
                correct = (y_hat.argmax(dim=-1) == labels).sum().item()
                print(f'epoch: {epoch+1}, correct: {correct}, total_num: {len(labels)}, accuracy: {correct / len(labels):.6f}, loss: {l.item()}')

    torch.save(net.state_dict(), './data/lstm_text_classification.pth')

if __name__ == '__main__':
    train_lstm()