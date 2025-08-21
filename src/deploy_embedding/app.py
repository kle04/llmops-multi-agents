from transformers import AutoTokenizer, AutoModel
import torch
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    # Use mean pooling
    embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()
    return {"embedding": embedding}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}