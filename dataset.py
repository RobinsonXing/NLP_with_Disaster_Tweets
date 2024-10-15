import pandas as pd
from typing import Literal

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch

class TweetSet(Dataset):
    def __init__(self, data, labels=None, max_len=256):
        """
        data: 经过处理的特征数据 (pandas DataFrame)
        labels: 标签列 (可选, 默认为 None)
        max_len: BERT 输入的最大长度
        """
        self.data = data
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])
        other_features = self.data.iloc[index].drop('text').values  # 其他特征
        other_features = other_features.astype(float)  # 强制转换为浮点数

        # BERT 分词
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 将其他特征转换为张量
        other_features = torch.tensor(other_features, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
            return input_ids, attention_mask, other_features, label
        else:
            return input_ids, attention_mask, other_features

def load_and_process_data(mode:Literal['train', 'test']):
    """
    处理数据集，将其转换为适合模型输入的格式
    """
    train_data = pd.read_csv('./dataset/train.csv').drop(columns=['id'])
    test_data = pd.read_csv('./dataset/test.csv').drop(columns=['id'])
    length = len(train_data)

    # 训练集和测试集的特征同时处理
    all_data = pd.concat([train_data.drop(columns=['target']), test_data], ignore_index=True)

    # 处理 keyword 列：独热编码，并填充空值为 'unknown'
    all_data['keyword'].fillna('unknown', inplace=True)
    keywords = pd.get_dummies(all_data, columns=['keyword'], prefix='kw')

    # 处理 location 列
    all_data['location'] = all_data['location'].fillna('unknown')
    all_data['location'] = all_data['location'].apply(lambda x: 'New York' if 'New York' in x else ('other' if x != 'unknown' else 'unknown'))
    locations = pd.get_dummies(all_data, columns=['location'], prefix='loc')

    # 提取对应特征
    texts = all_data['text']

    if mode == 'train':
        labels = train_data['target']
        return keywords.iloc[:length], locations.iloc[:length], texts.iloc[:length], labels
    else:
        return keywords.iloc[length:], locations.iloc[length:], texts.iloc[length:]