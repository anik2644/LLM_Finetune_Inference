
import os

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv
from utils.DeviceChoice import get_device
from utils.inference_utils.HelperMethodsInference import get_my_HF_HUB_model, get_my_local_model



load_dotenv()





def run_inference_hf_hub_my_model(question):
    """Run inference with the fine-tuned model."""

    model, tokenizer = get_my_HF_HUB_model()
    tokenizedInputs = tokenize_input(question, tokenizer)
    model_output = generate_output_from_model(tokenizedInputs, model, tokenizer)
    decoded_output = detokenize_output(tokenizer, model_output[0])

    return decoded_output


def run_inference_from_local(question):
    """Run inference with the fine-tuned model."""

    model, tokenizer = get_my_local_model()
    tokenizedInputs = tokenize_input(question, tokenizer)
    model_output = generate_output_from_model(tokenizedInputs, model, tokenizer)
    decoded_output = detokenize_output(tokenizer, model_output[0])

    return decoded_output




def tokenize_input(question, tokenizer):
    """Formats and tokenizes a single data example for the T5 model."""
    # Ensure this format matches the 'format_data' function used during training!
    context = question.get('Context', '')
    input_text = f"question: {str(question['Question'])} context: {context}"

    # Tokenize the input text
    inputs = tokenizer(
        input_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return inputs


def generate_output_from_model(inputs, model, tokenizer):
    """Generates the output sequence (answer) using the T5 model."""
    # Move inputs to the correct device


    device = get_device()
    # print(f"Using device: {device}")

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate the output sequence
    # T5 uses the 'generate' method for sequence-to-sequence inference
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=150,  # Max length for the generated answer
        num_beams=4,  # Beam search for higher quality output
        early_stopping=True,
        no_repeat_ngram_size=3,  # Discourage repetitive phrases
    )

    return outputs




def detokenize_output(tokenizer,output):
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    return decoded_output