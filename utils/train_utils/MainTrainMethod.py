from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import os
from utils.train_utils.HelperMethodsTrain import load_model_and_tokenizer_base_model
from utils.train_utils.TrainUtils import get_train_args, CheckpointMonitorCallback,analyze_checkpoint
from dotenv import load_dotenv
from huggingface_hub import login

# Replace with your token

import torch
# Custom callback for checkpoint monitoring
from transformers import TrainerCallback

load_dotenv()

def start_training(train_dataset, tokenized_train):
    token = os.getenv("HF_TOKEN")
    login(token)


    model , tokenizer =load_model_and_tokenizer_base_model();
    print(f"Training dataset size: {len(train_dataset)}")

    # Calculate steps per epoch for saving every 5 epochs
    batch_size = 4
    total_epochs = 30
    steps_per_epoch = len(train_dataset) // batch_size
    save_steps_every_5_epochs = steps_per_epoch * 5

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Save steps every 5 epochs: {save_steps_every_5_epochs}")

    # Enhanced training arguments

    training_args = get_train_args(total_epochs, batch_size, save_steps_every_5_epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        callbacks=[CheckpointMonitorCallback()]
    )



    # Initialize Trainer with callback

    # Function to check what's in a checkpoint


    # Check initial model size
    print("Initial model analysis:")
    initial_size = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {initial_size:,}")
    print(f"Model size (approx): {initial_size * 4 / (1024**2):.2f} MB (FP32)")

    # Main training with resume capability
    try:
        print("🚀 Starting training...")
        # Check if we have existing checkpoints
        if os.path.exists("./testing_checkpoint"):
            checkpoints = [d for d in os.listdir("./testing_checkpoint")
                          if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                latest_path = os.path.join("./testing_checkpoint", latest_checkpoint)
                print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
                analyze_checkpoint(latest_path)
                trainer.train(resume_from_checkpoint=latest_path)
            else:
                print( "at else condition")
                trainer.train()
        else:
            trainer.train()

    except Exception as e:
        print(f"❌ Training error: {e}")
        print("🔄 Starting fresh training...")
        trainer.train()

    # After training completion
    print("🎉 Training completed! Saving final model...")
    trainer.save_model("./testing_checkpoint")
    tokenizer.save_pretrained("./testing_checkpoint")

    # Push final model to Hub
    # model.push_to_hub("mhdank/testing_checkpoint", commit_message="Final trained model - 30 epochs")
    # tokenizer.push_to_hub("mhdank/testing_checkpoint", commit_message="Final tokenizer")

    print("✅ Training complete and model saved!")



