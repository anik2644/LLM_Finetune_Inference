from transformers import Trainer, TrainingArguments
from utils.train_utils.HelperMethodsTrain import load_model_and_tokenizer_base_model, save_model
from datasets import Dataset
import torch
from utils.train_utils.LoadData import get_data
from utils.train_utils.TokenizerUtils import format_data,tokenize_data
from utils.train_utils.MainTrainMethod import start_training
import os

# Set training parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
total_epochs = 30

# Load data (assumed from provided code)
train_data, test_data, val_data = get_data()

print(f" train data size; {len(train_data)}")
print(f" validation data size; {len(val_data)}")
print(f" test data size; {len(test_data)}")

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)
val_dataset = Dataset.from_list(val_data)


print("==============11111==================")
print(train_dataset)
print(val_dataset)


train_dataset = train_dataset.map(format_data)
test_dataset = test_dataset.map(format_data)
val_dataset = val_dataset.map(format_data)


print("================22222================")
print(train_dataset)

# Load model and tokenizer
model, tokenizer= load_model_and_tokenizer_base_model()


# Map the tokenizer function to the datasets
tokenized_train_dataset = train_dataset.map(lambda x: tokenize_data(x, tokenizer=tokenizer), batched=True)
tokenized_test_dataset = test_dataset.map(lambda x: tokenize_data(x, tokenizer=tokenizer), batched=True)
tokenized_val_dataset = val_dataset.map(lambda x: tokenize_data(x, tokenizer=tokenizer), batched=True)


print("================33333================")
print(tokenized_train_dataset)

start_training(train_dataset,tokenized_train_dataset)

# Save the trained model
# save_model(model, tknzr, "./final_model")
