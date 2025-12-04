"""
使用 huggingface 的 datasets 组件进行数据加载
"""
import torch
from datasets import load_dataset
from torch.utils import data
from transformers import BertTokenizer, BertModel
from torch import nn


def load_datasets_test():
    """
    split 支持的参数, Split splits.py 具体如下
        TRAIN = NamedSplit("train")
        TEST = NamedSplit("test")
        VALIDATION = NamedSplit("validation")
        ALL = NamedSplitAll() ------ 错误

    从测试来看, split 只能是 train

    """
    # load_dataset 函数将 path 参数解释为数据集标识符而非文件路径`
    dataset_train = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
    '''
    dataset_train: Dataset({
        features: ['label', 'review'],
        num_rows: 11987
    })
    '''
    print(f'dataset_train: {dataset_train}')
    '''
    {'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'review': ['很快，好吃，味道足，量大', '没有送水没有送水没有送水', '非常快，态度好。', '方便，快捷，味道可口，快递给力', '菜味道很棒！送餐很及时！', '今天师傅是不是手抖了，微辣格外辣！', '送餐快,态度也特别好,辛苦啦谢谢', '超级快就送到了，这么冷的天气骑士们辛苦了。谢谢你们。麻辣香锅依然很好吃。', '经过上次晚了2小时，这次超级快，20分钟就送到了……', '最后五分钟订的，卖家特别好接单了，谢谢。']}
    '''
    print(dataset_train[:10])

model_path = "E:\\study\\AI\\nlp_data\\model\\bert-base-chinese"

tokenize = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
bert_model = BertModel.from_pretrained(pretrained_model_name_or_path=model_path) # BertModel

def collate_fn_load_data(data_list):
    """
    data 的数据格式为
    [{'label': 0, 'review': '点了个帕尼尼，我去！这啥？怎么好意思叫帕尼尼？难吃死了。星巴克帕尼尼，难吃中的战斗机！'}, {'label': 0, 'review': '太慢了，口味对我来说是一般。'}]
    """
    # 从data中获取数据 和 标签数据
    labels = [record['label'] for record in data_list]
    sentences = [record['review'] for record in data_list]

    '''
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            list[TextInput],
            list[TextInputPair],
            list[PreTokenizedInput],
            list[PreTokenizedInputPair],
            list[EncodedInput],
            list[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, None] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    )
    '''
    batch_encode_data = tokenize.batch_encode_plus(batch_text_or_text_pairs=sentences, padding=True, truncation=True, max_length=50, return_tensors='pt', return_length=True)
    '''
    {'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]]), 'input_ids': tensor([[ 101, 2157, 7027,  782, 6963, 1391, 4638, 6656, 4007, 2692, 8024, 1282,
         1146,  102,    0,    0,    0,    0, ...   0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0]]), 'length': tensor([14, 27, 17, 50, 16, 25, 14, 19, 43, 14, 14, 11, 38, 21, 50,  7]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]])}
    '''
    # print(f'batch_encode_data: {batch_encode_data}')


    input_ids, token_type_ids, attention_mask = batch_encode_data['input_ids'], batch_encode_data['token_type_ids'], batch_encode_data['attention_mask']
    labels = torch.tensor(labels)
    return input_ids, token_type_ids, attention_mask, labels


# 对 torch.utils.data.dataloader 参数 collate_fn 使用方式进行测试
def data_loader_collate_fn_test():
    # 加载数据集
    dataset_train = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')

    # 创建数据迭代器 train_data_iter
    '''
    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[list], Iterable[list], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
    )
    '''
    # 基础参数
    batch_size = 16 # 余3

    data_train_iter = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_load_data, drop_last=True)

    for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(data_train_iter):
        # 0 torch.Size([16, 50]) torch.Size([16, 50]) torch.Size([16, 50]) torch.Size([16])
        print(idx, input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape)
        # print(f'input_ids: {input_ids}')
        # print(f'token_type_ids: {token_type_ids}')
        # print(f'attention_mask: {attention_mask}')
        # print(f'labels: {labels}')

        break

# 定义 经过 bert 预训练模型后, 经过的全连接层fc
class TextClassificationModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            '''
            BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                cross_attentions=encoder_outputs.cross_attentions,
            )
            '''
            bert_output = bert_model(input_ids, attention_mask, token_type_ids)

        # NOTE: 每个句子都有[CLS], 使用CLS的特征代表整个句子的信息
        output = self.fc(bert_output.last_hidden_state[:, 0])
        return output

def text_classification_test():
    # 加载数据集
    dataset_train = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
    # 构造data_iter
    data_train_iter = data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn_load_data, drop_last=True)

    bert_fc_model = TextClassificationModule(in_features=768, out_features=2)

    for i, (input_ids, toke_type_ids, attention_mask, labels) in enumerate(data_train_iter):
        y_hat = bert_fc_model(input_ids, toke_type_ids, attention_mask)
        print(f'y_hat.shape: {y_hat.shape}, y_hat: {y_hat}') # y_hat.shape: torch.Size([16, 2])

        break

def train_model():
    # 加载数据
    train_dataset = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
    # 定义训练需要使用的模型和相关方法
    net = TextClassificationModule(in_features=768, out_features=2) # 模型
    loss = nn.CrossEntropyLoss() # 损失函数
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=1e-4)
    # 让预训练模型的权重不更新
    for param in bert_model.parameters():
        param.requires_grad_ = False

    # 进行训练
    epochs, batch_size = 3, 16
    for epoch in range(epochs):
        data_train_iter = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_load_data, drop_last=True)
        for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(data_train_iter):
            y_hat = net(input_ids, token_type_ids, attention_mask)
            l = loss(y_hat, labels) # 计算损失, y_hat、labels 差一个维度

            # 梯度清零
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # 每10个迭代计算一次精度
            if idx % 10 == 0:
                y_argmax = y_hat.argmax(dim=-1)
                correct = (y_argmax == labels).sum().item()
                print(f'epoch: {epoch+1}, correct: {correct}, accuracy: {correct / batch_size:.6f}, loss: {l.item():.6f}')

    torch.save(net.state_dict(), './data/text_classification.pth') # 保存模型


def evaluate_model():
    # 加载数据
    test_dataset = load_dataset(path='csv', data_files='./data/text_classification.csv', split='train')
    # 声明模型
    net = TextClassificationModule(in_features=768, out_features=2)
    net.load_state_dict(torch.load('./data/text_classification.pth', weights_only=True)) # 加载模型参数

    net.eval()

    total_num = 0
    correct_num = 0
    test_iter = data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_load_data, drop_last=True)
    for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(test_iter):
        with torch.no_grad():
            y_pred = net(input_ids, token_type_ids, attention_mask)
        y_argmax = y_pred.argmax(dim=-1)
        correct = (y_argmax == labels).sum().item()

        total_num += len(labels)
        correct_num += correct

        if idx % 10 == 0:
            print(f'correct: {correct_num}, accuracy: {correct_num / total_num:.6f}')

    print(f'最终准确率: {correct_num / total_num:.6f}')

if __name__ == '__main__':
    # NOTE: 这里缺少基础数据处理逻辑, 可以将数据中类似 '" 这样的符号去掉, 并且对字符和特殊符合连接的问题, 应该用空格隔开
    # load_datasets_test()
    # data_loader_collate_fn_test()
    # text_classification_test()
    train_model()
    # evaluate_model()

    pass
