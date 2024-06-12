from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import json
from datasets.openweb import OpenWebDataSet
import torch

# Step 1: Preprocess the dataset 
def create_new_dataset(dataset, tokenizer, ideal_token_size_top, ideal_token_size_low, ideal_dataset_size):

    ## ...
    dataset_size = 0
    new_dataset = []

    ## ...
    shuffled_indices = list(range(len(dataset["train"])))
    random.seed(42)
    random.shuffle(shuffled_indices)

    for i in tqdm(shuffled_indices):
        
        ## ...
        text = dataset["train"][i]["text"]
        tokenized = tokenizer(text)
        length = len(tokenized["input_ids"])
        
        ## ...
        if length <= ideal_token_size_top and length >= ideal_token_size_low:
            new_dataset.append(tokenized)
            dataset_size += 1

            ## ...
            if dataset_size % 100_000 == 0:
                print(f"Dataset size: {dataset_size}")

            ## ... 
            if dataset_size == ideal_dataset_size:
                break
    

    return new_dataset


def split_train_val(dataset, ratio_eval):
    
    ## Generate shuffled indices
    shuffled_indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(shuffled_indices)

    ## Calculate the number of entries to put in the validation set
    val_indices = int(ratio_eval * len(shuffled_indices))

    ## Use the shuffled indices to select data for train and validation sets
    train_indices = shuffled_indices[val_indices:]
    validation_indices = shuffled_indices[:val_indices]

    ## Create the train and validation sets
    train = [dataset[i] for i in train_indices]
    validation = [dataset[i] for i in validation_indices]

    ## Create the new Dataset Dict
    dataset_dict = DatasetDict({"train": train, "validation": validation})

    return dataset_dict

def check_lengths(dataset):
    
    lengths = []

    for i in tqdm(range(len(dataset))):

        if len(dataset[i]["input_ids"]) > 256:
            length = len(dataset[i]["input_ids"])
            length_add = (i, length)
            lengths.append(length_add)

    return lengths

# Load & copy dataset from HuggingFace
if __name__ == "_main_":
    MODEL_CHECKPOINT = "bert-base-cased"

    ## ...
    IDEAL_DATASET_SIZE = 1_000_000
    IDEAL_TOKEN_SIZE_TOP = 512
    IDEAL_TOKEN_SIZE_LOW = 100

    ## ...
    RATIO_EVAL = 0.10


    tokenizer = BertTokenizer.from_pretrained(MODEL_CHECKPOINT)

    dataset = load_dataset("Skylion007/openwebtext")
    filtered_dataset = dataset.copy()

    # Length of the original dataset
    len_original = len(dataset["train"])

    # Filter dataset to remove longer texts
    filtered_dataset["train"] = dataset["train"].filter(lambda example: len(example["text"]) <= 2000)

    # Create new dataset
    new_dataset = create_new_dataset(filtered_dataset, tokenizer, IDEAL_TOKEN_SIZE_TOP, IDEAL_TOKEN_SIZE_LOW, IDEAL_DATASET_SIZE)

    # Print reduction in datasize
    len_new = len(new_dataset)
    print(f"Original dataset size: {len_original}")
    print(f"New dataset size: {len_new}")
    print(f"Reduction in dataset size: {((1 - (len_original - len_new) / len_original)) * 100}%")

    # Check if the new dataset has any text longer than 256 tokens
    lengths = check_lengths(new_dataset)
    print(lengths)

    # Split in train and val
    splitted_dataset = split_train_val(new_dataset, RATIO_EVAL)

    # Step 2: Save the dataset in a format suitable for loading with PyTorch
    # Save dataset as a list of dictionaries

    from tqdm import tqdm

    train_data = [{"input_ids": data["input_ids"], "attention_mask": data["attention_mask"]} for data in splitted_dataset["train"]]
    val_data = [{"input_ids": data["input_ids"], "attention_mask": data["attention_mask"]} for data in splitted_dataset["validation"]]

    with open("data/train_data.json", 'w') as f:
        json.dump(train_data, f)

    with open("data/val_data.json", 'w') as f:
        json.dump(val_data, f)


    # create dataset dump
    train_dataset = OpenWebDataSet("data/train_data.json")
    val_dataset = OpenWebDataSet("data/val_data.json")

    dataset = {
        "trainset": train_dataset,
        "testset": val_dataset,
    }

    torch.save(dataset, "data/openweb.pt")
