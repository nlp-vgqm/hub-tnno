"""
使用 rnn 进行文本分类
"""
import torch
from torch import nn
from torch.utils import data
from transformers import BertTokenizer, BertModel
from datasets import load_dataset

model_path = "E:\\study\\AI\\nlp_data\\model\\bert-base-chinese"
tokenize = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=model_path)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embed = bert_model.embeddings.word_embeddings # 使用bert模型进行张量化的, 这里使用它的 word_embedding 词嵌入层, segment_embedding、position_embedding 暂时不用
        self.net = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X, hidden):
        """
        需要将 X 转为 模型认识的维度 [16,, 50] -> [16, 50, 768]
        :param X:
        :param hidden:
        :return:
        """
        embed_X = self.embed(X)
        rnn_X, hidden = self.net(embed_X, hidden) # [16, 50, 768], [1, 16, 768]
        output = self.fc(rnn_X[:, -1]) # [16, 768] -> [16, 2] 取最后一个时间步的输出作为特征
        return output, hidden # [16, 2]

    def init_hidden(self, batchSize):
        return torch.zeros(size=(self.num_layers, batchSize, self.hidden_size))

class GRUDataset(data.Dataset):
    def __init__(self):
        self.dataset = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
        self.length = len(self.dataset)

    def __getitem__(self, idx):
        """
        为了简化 数据操作, 这里使用 bert-base-chinese 一样的方式进行数据加载
        :param idx:
        :return:
        """
        review = self.dataset[idx]['review']
        label = self.dataset[idx]['label']

        return review, label

    def __len__(self):
        return self.length

def gru_collate_fn(records):
    """
    数据格式
    [('很快，好吃，味道足，量大', 1), ('很快，好吃，味道足，量大', 1)]
    :param records:
    :return:
    """
    reviews = [record[0] for record in records]
    labels = [record[1] for record in records]

    # 对数据进行张量化处理
    encode_input = tokenize.batch_encode_plus(batch_text_or_text_pairs=reviews, max_length=max_len, padding=True, truncation=True, return_tensors='pt', return_length=True)
    input_ids = encode_input['input_ids']
    labels = torch.LongTensor(labels)

    return input_ids, labels

max_len = 50
batch_size, lr, epochs = 16, 1e-3, 3
def train_gru_model():
    # 加载数据
    train_dataset = GRUDataset()

    net = GRUClassifier(input_size=768, hidden_size=768, output_size=2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)

    total_num = 0
    correct_num = 0
    for epoch in range(epochs):
        data_iter = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gru_collate_fn, drop_last=True)
        for i, (input_ids, labels) in enumerate(data_iter):
            y_hat, hidden = net(input_ids, net.init_hidden(batch_size)) # y_hat [16, 2]
            l = loss(y_hat, labels)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_num += len(labels)
            correct_num += (y_hat.argmax(dim=-1) == labels).sum().item()
            if i % 100 == 0: # NOTE: 这里使用的是累计统计, 和其它3个不同
                print(f'epoch: {epoch+1}, correct: {correct_num}, total_num: {total_num}, accuracy: {correct_num / total_num:.6f}, loss: {l.item()}')

    torch.save(net.state_dict(), './data/gru_text_classification.pth')

# 注意: 本训练没有对文件进行分词操作, 来得到 word2idx, idx2word 等相关信息, 而是使用 bert 的 word_embedding, 如果需要分词, 可以使用 jieba, 然后自定义 embedding 层
if __name__ == '__main__':
    train_gru_model()