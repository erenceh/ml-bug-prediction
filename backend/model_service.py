import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "erenceh/ml-bug-priority"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=MODEL_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, token=MODEL_TOKEN
).to(device)


def predict_priority(title: str, description: str):
    text = f"{title} {description}"
    inputs = {
        k: v.to(device)
        for k, v in tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=296
        ).items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()

    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()

    return predicted_class, confidence, probs
