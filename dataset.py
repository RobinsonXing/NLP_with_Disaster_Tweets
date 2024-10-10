from typing import Literal
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import Dataset

class TweetSet(Dataset):
    def __init__(self, mode:Literal['train', 'test'], max_len=128):
        
        # 读取训练集和测试集
        train_data = pd.read_csv('./dataset/train.csv')
        test_data = pd.read_csv('./dataset/test.csv')
        length = len(train_data)

        # 将训练集和测试集特征合并，去除 'id' 和 'location' 列
        train_features = train_data.drop(columns=['target'])
        all_features = pd.concat([train_features, test_data], ignore_index=True)
        all_features = all_features.drop(columns=['id', 'location'])

        # 填充 keyword 中的空值并进行独热编码
        all_features['keyword'].fillna('unknown', inplace=True)
        all_features = pd.get_dummies(all_features, columns=['keyword'], prefix='kw')

        # 分离训练和测试特征
        train_features = all_features.iloc[:length]
        test_features = all_features.iloc[length:]
        if mode == 'train':
            self.features = train_features
            self.labels = train_data['target']
        else:
            self.features = test_features
            self.labels = None
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        # 获取文本和其他特征
        text = self.features.iloc[index]['text']
        other_features = self.features.iloc[index].drop('text').values
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 获取 BERT 的 input_ids 和 attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 转换其他特征为 tensor
        other_features = torch.tensor(other_features, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
            return input_ids, attention_mask, other_features, label
        else:
            return input_ids, attention_mask, other_features