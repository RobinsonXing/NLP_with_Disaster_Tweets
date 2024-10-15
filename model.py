import torch
import torch.nn as nn
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self, num_keyword_features, num_location_features, num_labels=1):
        super(MultimodalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 数值特征处理
        self.fc_keywords = nn.Linear(num_keyword_features, 64)
        self.fc_location = nn.Linear(num_location_features, 32)
        
        # 融合后的全连接层
        self.fc_combined = nn.Linear(self.bert.config.hidden_size + 64 + 32, 64)
        self.classifier = nn.Linear(64, num_labels)
    
    def forward(self, input_ids, attention_mask, keyword_features, location_features):
        # BERT 处理文本
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled_output = bert_output.pooler_output

        # 分别处理数值特征
        keyword_output = torch.relu(self.fc_keywords(keyword_features))
        location_output = torch.relu(self.fc_location(location_features))

        # 拼接 BERT 输出和数值特征
        combined_output = torch.cat((bert_pooled_output, keyword_output, location_output), dim=1)

        # 融合后通过全连接层
        combined_output = torch.relu(self.fc_combined(combined_output))
        logits = self.classifier(combined_output)

        return logits