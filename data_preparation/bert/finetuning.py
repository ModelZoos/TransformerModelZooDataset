# %% [markdown]
# # Data Preparation

# %%
from transformers import BertTokenizer 
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch.nn.functional as F

from datasets.sst import SSTDataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset('glue', 'sst2')

## Extract the train, validation and test sets
raw_sst2_train = dataset['train']
raw_sst2_val = dataset['validation']

raw_sst2_train = pd.DataFrame(raw_sst2_train)
raw_sst2_val = pd.DataFrame(raw_sst2_val)

raw_sst2_trainval = pd.concat([raw_sst2_train, raw_sst2_val])
raw_sst2_trainval = raw_sst2_trainval.drop_duplicates(subset='sentence', keep=False)


# ## Data Preprocessing
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []

for txt in raw_sst2_trainval.sentence:
  tokens = tokenizer.encode(txt, max_length=512, truncation=True)
  token_lens.append(len(tokens))

df_train, df_test = train_test_split(raw_sst2_trainval, test_size=0.3, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.66, random_state=RANDOM_SEED)

## Calculate distribution of train, val and test
print(len(df_train) / len(raw_sst2_trainval))
print(len(df_val) / len(raw_sst2_trainval))
print(len(df_test) / len(raw_sst2_trainval))


## Save dataframes to json
df_train.to_json('data/train.json', orient='records', lines=True)
df_val.to_json('data/val.json', orient='records', lines=True)
df_test.to_json('data/test.json', orient='records', lines=True)

def create_ds(df, tokenizer, max_len, batch_size):
  ds = SSTDataset(sentences=df.sentence.to_numpy(),
                        labels=df.label.to_numpy(),
                        tokenizer=tokenizer,
                        max_len=max_len)

  return ds


train_ds = create_ds(df_train, tokenizer, 80, 64)
test_ds = create_ds(df_test, tokenizer, 80, 64)

dataset = {
    "trainset": train_ds,
    "testset": test_ds
}

torch.save(dataset, "data/sst.pt")
