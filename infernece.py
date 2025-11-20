from transformers import T5ForConditionalGeneration, T5Tokenizer

def run_inference_base(question, model, tokenizer):
    """Run inference with the base model."""
    input_text = f"question: {question} context: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def run_inference(question, model, tokenizer):
    """Run inference with the fine-tuned model."""
    input_text = f"question: {question} context: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
