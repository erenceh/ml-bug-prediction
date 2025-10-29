import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

MODEL_ID = "erenceh/ml-bug-priority"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
pipe = pipeline("text-classification", model=MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_priority(title: str, description: str):
    text = f"{title} {description}"
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=296
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    predicted_class = torch.argmax(probs).item()
    confidence = float(probs[predicted_class].item())
    probs_dict = {i: float(probs[i]) for i in range(len(probs))}

    return predicted_class, confidence, probs_dict
