from fastapi import FastAPI
from pydantic import BaseModel
from utils.inference_utils.MainInferenceMethod import run_inference_from_local
import uvicorn

# Define the request body model
class InferenceRequest(BaseModel):
    Question: str
    Context: str = ""

# Create the FastAPI app
app = FastAPI()

@app.post("/predict/")
def predict(request: InferenceRequest):
    """
    Runs inference on the provided question and context.
    """
    example = {"Question": request.Question, "Context": request.Context}
    predicted_answer = run_inference_from_local(example)
    return {"Predicted Answer": predicted_answer}

@app.get("/")
def read_root():
    return {"message": "T5 Inference Server is running. Use the /predict/ endpoint to get predictions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
