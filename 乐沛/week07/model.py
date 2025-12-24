"""
    定义神经网络模型结构
"""

import torch
import torch.nn as nn
from transformers import BertModel


#定义RNN模型
class TorchModel_Rnn(nn.Module):
    def __init__(self, output_size, sentence_length, input_size, pooling):
        super(TorchModel_Rnn, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size, 0)
        self.layer = nn.RNN(output_size, output_size, bias=False, batch_first=True)
        self.linear = nn.Linear(output_size, 1)
        if pooling == 'avg':
            self.pool = nn.AvgPool1d(sentence_length)
        else:
            self.pool = nn.MaxPool1d(sentence_length)
        self.loss = nn.functional.mse_loss
        self.activation = torch.sigmoid

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.layer(x)

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()

        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


#定义BERT模型
class TorchModel_BERT(nn.Module):
    def __init__(self, output_size, sentence_length, input_size):
        super(TorchModel_BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.liner = nn.Linear(input_size, 1)
        self.activation = torch.sigmoid
        self.dropout = nn.Dropout(0.5)
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)

        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

#定义cnn模型
class TorchModel_Cnn(nn.Module):
    def __init__(self, output_size, sentence_length, input_size, pooling):
        super(TorchModel_Cnn, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size, 0)
        self.cnn = nn.Conv1d(
            in_channels=output_size,
            out_channels=output_size,  # 输出通道数与输入相同
            kernel_size=3,
            padding=1,  # 保持序列长度不变
            bias=False
        )
        self.linear = nn.Linear(output_size, 1)
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        self.loss = nn.functional.mse_loss
        self.activation = torch.sigmoid

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#定义lstm模型
class TorchModel_Lstm(nn.Module):
    def __init__(self, output_size, sentence_length, input_size, pooling):
        super(TorchModel_Lstm, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size, 0)
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=output_size,  # 输出通道数与输入相同
            bias=False,
            batch_first=True
        )
        self.linear = nn.Linear(output_size, 1)
        if pooling == 'avg':
            self.pool = nn.AvgPool1d(sentence_length)
        else:
            self.pool = nn.MaxPool1d(sentence_length)
        self.loss = nn.functional.mse_loss
        self.activation = torch.sigmoid

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
