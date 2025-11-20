import os
from transformers import TrainingArguments, TrainerCallback
import torch


def get_train_args(total_epochs,batch_size,save_steps_every_5_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    training_args = TrainingArguments(
        output_dir="./testing_checkpoint",
        num_train_epochs=total_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=3e-4,
        weight_decay=0.01,

        # Checkpointing configuration
        save_strategy="steps",
        save_steps=save_steps_every_5_epochs,
        save_total_limit=3,  # Keep checkpoints for every 5 epochs (30/5=6)

        # Hugging Face Hub configuration
        push_to_hub=False,
        hub_model_id="mhdank/testing_checkpoint",  # Using your repo
        hub_strategy="checkpoint",

        # Other settings
        logging_steps=20,
        report_to="none",
        dataloader_pin_memory=False,
        load_best_model_at_end=False,
        remove_unused_columns=True, # Changed from False to True

        # Better checkpoint naming
        overwrite_output_dir=True,

    )

    return training_args



class CheckpointMonitorCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"

        if os.path.exists(checkpoint_path):
            # Calculate checkpoint size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(checkpoint_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)

            # Convert to MB
            size_mb = total_size / (1024 * 1024)

            # Save the model directly to the base folder



            # Calculate current epoch
            current_epoch = state.epoch
            print(f"✅ Checkpoint saved at epoch {current_epoch:.1f}")
            print(f"📁 Checkpoint path: {checkpoint_path}")
            print(f"💾 Checkpoint size: {size_mb:.2f} MB")
            print(f"🔢 Global step: {state.global_step}")
            print("---")


def analyze_checkpoint(checkpoint_path):
    """Analyze what's stored in a checkpoint"""
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print("❌ Checkpoint directory doesn't exist")
        return

    # List all files in checkpoint
    files = []
    for dirpath, dirnames, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            files.append((filename, file_size))

    print("📄 Files in checkpoint:")
    for filename, size in files:
        print(f"   {filename}: {size:.2f} MB")

    total_size = sum(size for _, size in files)
    print(f"📦 Total checkpoint size: {total_size:.2f} MB")

    # Check if it's a full model checkpoint
    has_pytorch_model = any('pytorch_model.bin' in f[0] for f in files)
    has_training_args = any('training_args.bin' in f[0] for f in files)
    has_optimizer = any('optimizer.pt' in f[0] for f in files)
    has_scheduler = any('scheduler.pt' in f[0] for f in files)

    print("\n📊 Checkpoint contents:")
    print(f"   ✅ Model weights: {has_pytorch_model}")
    print(f"   ✅ Training arguments: {has_training_args}")
    print(f"   ✅ Optimizer state: {has_optimizer}")
    print(f"   ✅ Scheduler state: {has_scheduler}")

    if has_pytorch_model and has_training_args and has_optimizer and has_scheduler:
        print("\n🎯 This is a FULL checkpoint (weights + training state)")
        print("   Can resume training exactly where it left off")
    else:
        print("\n⚠️  This is a PARTIAL checkpoint")
