import torch
import argparse
import wandb


from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import TweetSet

# 参数设定
def parse_args():
    parser = argparse.ArgumentParser(description="BERT fine-tuning with cross-validation")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs")
    parser.add_argument("--max_len", type=int, default=128, 
                        help="Maximum sequence length for BERT")
    parser.add_argument("--kfold", type=int, default=5, 
                        help="Number of splits for cross-validation")
    parser.add_argument("--project_name", type=str, default="bert_cross_validation", 
                        help="WandB project name")
    return parser.parse_args()

def train():

    # 初始化参数
    arg = parse_args()

    # 初始化wandb
    wandb.init(project=arg.project_name)

    # 假设 TweetSet 已经定义好，使用它来加载训练集和验证集
    train_dataset = TweetSet(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 初始化 BERT 模型
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)  # 二分类任务

    # 设置设备 (GPU/CPU)
    device = torch.device(f'cuda:{arg.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # 设置训练的 epoch 数和训练步骤
    epochs = 3
    total_steps = len(train_loader) * epochs

    # 使用线性学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    # 损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 开始训练循环
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids, attention_mask, other_features, labels = batch

            # 将输入移动到 GPU/CPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).unsqueeze(1)  # 二分类任务

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

        # 输出每轮的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"Average loss: {avg_loss:.4f}")

    # 微调结束后保存模型
    model.save_pretrained('./fine_tuned_bert')

if __name__ == '__main__':
    main()
