import argparse
import wandb
import pandas as pd

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import TweetSet, load_and_process_data
from model import MultimodalModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multimodal BERT model")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--wandb_run', type=str, default='run1', help='wandb run name')
    parser.add_argument('--wandb_project', type=str, default='bert-multimodal', help='wandb project name')
    return parser.parse_args()

def train(args):
    # 初始化 wandb
    # wandb.init(project=args.wandb_project)
    # wandb.run.name = args.wandb_run
    # wandb.run.save()
    
    # 加载并处理数据
    keyword_data, location_data, text_data, labels = load_and_process_data(mode='train')
    dataset = TweetSet(pd.concat([keyword_data, location_data, text_data], axis=1), labels)

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        
        # 创建训练集和验证集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # 初始化模型
        model = MultimodalModel(num_keyword_features=keyword_data.shape[1], 
                                num_location_features=location_data.shape[1])
        model.to(device)
        
        # 定义优化器和学习率调度
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        criterion = torch.nn.BCEWithLogitsLoss()

        # 开始训练
        model.train()
        for epoch in range(args.epochs):
            total_train_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, keyword_features, location_features, labels = [x.to(device) for x in batch]
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, keyword_features, location_features)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            # `wandb.log({"train_loss": avg_train_loss})

            # 验证
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, keyword_features, location_features, labels = [x.to(device) for x in batch]
                    
                    outputs = model(input_ids, attention_mask, keyword_features, location_features)
                    loss = criterion(outputs.squeeze(), labels.float())
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(avg_train_loss)
            # wandb.log({"val_loss": avg_val_loss})

if __name__ == "__main__":
    args = parse_args()
    train(args)