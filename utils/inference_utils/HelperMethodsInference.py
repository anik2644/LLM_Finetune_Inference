import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from utils.DeviceChoice import get_device

load_dotenv()






def get_my_model():
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




