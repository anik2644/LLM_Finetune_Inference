import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer_base_model():
    model_name = "t5-base"
    """Load model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def save_model(model, tokenizer, output_dir):
    """Save model and tokenizer to output directory."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
