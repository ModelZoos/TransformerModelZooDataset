import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertModel

class MaskedLanguageModel(nn.Module):
    def __init__(self, case):
        super(MaskedLanguageModel, self).__init__()
        
        ## Differentiate btw. continued and scratch (I expected both to be same, but somehow it does not match)
        if case == "continued":
            base_model = "bert-base-cased"
        
        elif case == "scratch":
            base_model = "bert-base-uncased"
            
        self.bert = BertForMaskedLM.from_pretrained(base_model)
            
        # Reinitialize the model
        if case == "scratch":
            self.bert.apply(self.init_weights)

    ## Function to initialize weights
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        prediction_scores = bert_output.logits
        
        return prediction_scores
    
    def calculate_mlm_scores(outputs, labels):
        
        ## ...
        logits = outputs
        
        ## ...
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        ## ...
        predictions = torch.argmax(probabilities, dim=-1)
        
        ## ...
        mask = (labels != -100)
        
        ## ...
        correct = (predictions == labels) * mask

        ## ...
        correct = correct.sum().item()
        total = mask.sum().item()

        return correct, total