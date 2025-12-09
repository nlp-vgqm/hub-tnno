"""
    定义评价指标，每轮训练后做评测
"""
from loader import *

#计算rnn模型的准确率
def eval_model(model, data_path, test_size, random_state, shuffle, texts):
    model.eval()
    train_vocab, test_vocab = datacsv(data_path, test_size, random_state, shuffle)
    x, y = datatrain_rnn(test_vocab, text_to_vocab(texts), text_long(texts))
    print(f"本次预测中正样本{sum(y)}个，负样本{len(y) - sum(y)}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            elif float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            else:
                wrong += 1
    print(f"预测正确个数：{correct}，正确率：{correct/(correct + wrong)}")
    return correct/(correct + wrong)

#计算bert模型的准确率
def eval_bert(model, data_path, test_size, random_state, shuffle, texts):
    model.eval()
    train_vocab, test_vocab = datacsv(data_path, test_size, random_state, shuffle)
    x, y = datatrain_rnn(test_vocab, text_to_vocab(texts), 128)
    print(f"本次预测中正样本{sum(y)}个，负样本{len(y) - sum(y)}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            elif float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            else:
                wrong += 1
    print(f"预测正确个数：{correct}，正确率：{correct/(correct + wrong)}")
    return correct/(correct + wrong)
