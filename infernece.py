from utils.inference_utils.SampleData import get_test_dataframe
from datasets import Dataset
from utils.inference_utils.MainInferenceMethod import run_inference_hf_hub_my_model ,run_inference_from_local
from utils.DeviceChoice import get_device


DEVICE = get_device()
print(f"Using device: {DEVICE}")




test_df = get_test_dataframe()
print(test_df)

# Convert to Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df)
print(f"Test data loaded: {len(test_dataset)} samples.")

predictions = []



for i, example in enumerate(test_dataset):
    print(f"Processing sample {i + 1}/{len(test_dataset)}...")


    # 2. Generate the prediction
    # predicted_answer = run_inference_hf_hub_my_model(example)
    predicted_answer = run_inference_from_local(example)

    # 3. Store the result
    predictions.append({
        "Question": example["Question"],
        "Context": example.get("Context", ""),
        "Predicted Answer": predicted_answer,
    })

    # Optional: Print the result for immediate inspection
    print(f"   Q: {example['Question']}")
    print(f"   A: {predicted_answer}\n")
