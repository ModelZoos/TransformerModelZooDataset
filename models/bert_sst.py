import torch
import torch.nn as nn
from transformers import BertModel


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, case, pretrained_model_path=None):
        super(SentimentClassifier, self).__init__()
        
        ## Differentiate btw. continued and scratch (I expected both to be same, but somehow it does not match)
        if case == "continued":
            base_model = "bert-base-cased"
        
        elif case == "scratch":
            base_model = "bert-base-uncased"
            
        self.bert = BertModel.from_pretrained(base_model)

        if pretrained_model_path is not None:
            self.load_partial_state_dict(pretrained_model_path)
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes))

        # Initialize MLP layers with Kaiming (He) initialization
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def load_partial_state_dict(self, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        # Remove the classification head weights
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classification_head')}
        
        self.load_state_dict(state_dict, strict=False)

    def reset_classifier(self, n_classes):
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes))

        # Initialize MLP layers with Kaiming (He) initialization
        for m in self.classification_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
        

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        return self.classification_head(pooled_output)
