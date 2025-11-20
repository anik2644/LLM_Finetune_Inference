import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from utils.DeviceChoice import get_device

load_dotenv()






def get_my_HF_HUB_model():
    # Load the fine-tuned tokenizer
    tokenizer = T5Tokenizer.from_pretrained(os.getenv("HUB_REPO_ID"))
    # Example usage
    DEVICE = get_device()
    # print(f"Using device: {DEVICE}")

    # Load the model weights. The 'from_pretrained' method will automatically
    # download the latest (best) model files from the Hugging Face Hub.
    model = T5ForConditionalGeneration.from_pretrained(os.getenv("HUB_REPO_ID"))
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode (crucial for inference!)
    return model, tokenizer


from transformers import T5ForConditionalGeneration, T5Tokenizer
import os


def get_my_local_model():
    # Path to the local checkpoint directory
    local_model_path = os.getenv("LOCAL_MODEL_PATH") # Update the path as needed

    # Load the fine-tuned tokenizer from the local path
    tokenizer = T5Tokenizer.from_pretrained(local_model_path)

    # Example usage
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load the model weights from the local directory
    model = T5ForConditionalGeneration.from_pretrained(local_model_path)

    # Move the model to the appropriate device (CPU or GPU)
    model.to(DEVICE)

    # Set the model to evaluation mode (important for inference)
    model.eval()

    return model, tokenizer



