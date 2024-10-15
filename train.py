import numpy as np
import pandas as pd
import argparse
import wandb
import random
from tqdm import tqdm
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

# 参数设定
def parse_args():
    parser = argparse.ArgumentParser(description="BERT fine-tuning with cross-validation")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--kfold", type=int, default=5)
    return parser.parse_args()

# 设置随机种子以确保结果的可重复性
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据集
class TweetSet(Dataset):
    def __init__(self, data, labels=None, max_len=128):
        self.data = data
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
       
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        other_features = self.data.iloc[index].drop('text').values
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
        other_features = torch.tensor(other_features, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)
            return input_ids, attention_mask, other_features, label
        else:
            return input_ids, attention_mask, other_features

# 训练
def train():
    # 初始化
    args = parse_args()
    wandb.init(project='Disaster_Tweets_BERT', config=args)
    wandb.run.save()

    # 读取并处理数据
    train_data = pd.read_csv('./dataset/train.csv')
    labels = train_data['target']
    train_features = train_data.drop(columns=['id', 'location', 'target'])
    train_features['keyword'].fillna('unknown', inplace=True)
    train_features = pd.get_dummies(train_features, columns=['keyword'], prefix='kw')

    # 设置设备 (GPU/CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 交叉验证循环
    fold_index = 1
    kf = KFold(n_splits=args.kfold, shuffle=True)
    for train_index, valid_index in kf.split(train_features):
        print(f'Fold {fold_index}')
        fold_index += 1

        # 根据索引划分训练集和验证集
        train_fold_features = train_features.iloc[train_index]
        val_fold_features = train_features.iloc[valid_index]
        train_fold_labels = labels.iloc[train_index]
        val_fold_labels = labels.iloc[valid_index]

 # 使用 TweetSet 创建训练集和验证集
        train_dataset = TweetSet(data=train_fold_features, labels=train_fold_labels, max_len=args.max_len)
        val_dataset = TweetSet(data=val_fold_features, labels=val_fold_labels, max_len=args.max_len)

        # 定义 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 初始化 BERT 模型
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        model.to(device)

        # 定义优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # 损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # 记录到 WandB
        wandb.watch(model)

        # 开始训练
        model.train()
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            total_loss = 0

            for batch in tqdm(train_loader):
                input_ids, attention_mask, other_features, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device).unsqueeze(1)

                # 前向传播
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                logits = outputs.logits

                # 计算损失
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数和学习率
                optimizer.step()
                scheduler.step()

                # 清空梯度
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"Average loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss})

        # 在验证集上评估模型
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, other_features, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss})

    # 结束 WandB 记录
    wandb.finish()

if __name__ == '__main__':
    train()
