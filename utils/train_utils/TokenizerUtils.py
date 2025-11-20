from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from utils.train_utils.LoadData import get_data



def format_data(example):
    input_text = f"question: {str(example['Question'])} context: "
    target_text = str(example["Answer"])

    # # Debug print to check the format
    # print(f"Input text example: {input_text}")
    # print(f"Target text example: {target_text}")

    return {
        "input_text": input_text,
        "target_text": target_text
    }


def tokenize_data(batch,tokenizer):
    # Tokenize input and target text
    input_enc = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    target_enc = tokenizer(
        batch["target_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Ensure correct structure
    # print(f"Tokenized input_enc: {input_enc['input_ids'][:5]}")  # Show a sample of tokenized input
    # print(f"Tokenized target_enc: {target_enc['input_ids'][:5]}")  # Show a sample of tokenized target

    input_enc["labels"] = target_enc["input_ids"]
    return input_enc
