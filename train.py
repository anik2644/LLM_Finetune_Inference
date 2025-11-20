from transformers import Trainer, TrainingArguments
from utils.train_utils import load_model_and_tokenizer, save_model
from datasets import Dataset
import os

# Set training parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
total_epochs = 30

# Load data (assumed from provided code)
train_data, test_data, val_data = load_data()  # Implement load_data based on file

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)
val_dataset = Dataset.from_list(val_data)

# Tokenize data
def tokenize_data(batch):
    input_enc = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=256)
    target_enc = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=256)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)
val_dataset = val_dataset.map(tokenize_data, batched=True)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Training arguments
training_args = TrainingArguments(
    output_dir="./output_model",
    num_train_epochs=total_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=3e-4,
    weight_decay=0.01,
    save_steps=1000,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Save the trained model
save_model(model, tokenizer, "./final_model")
