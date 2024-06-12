import json
import torch
from torch.utils.data import Dataset, DataLoader

class OpenWebDataSet(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        return {"input_ids": input_ids, "attention_mask": attention_mask}