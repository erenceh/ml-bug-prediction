import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import json
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("ml/models/bert", device_map="auto")
model = AutoModelForSequenceClassification.from_pretrained("ml/models/bert")
model.to(device)
model.eval()

with open("ml/models/bert/label_map.json", "r") as f:
  label_map = json.load(f)
  
label_map = {0: "Critical", 1: "High", 2: "Medium", 3: "Low", 4: "Trivial"}

st.title("Bug Priority Predictor")
text = st.text_area("Paste issue title + description")

if st.button("Predict"):
  if text.strip() == "":
    st.warning("Please enter some text!")
  else:
    inputs = tokenizer(
      text,
      truncation=True,
      max_length=296,
      padding="max_length",
      return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
      outputs = model(**inputs)
      logits = outputs.logits
      
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_index = probabilities.argmax(dim=-1).item()
    predicted_label = label_map[predicted_index]
  
    class_probs = {label_map[i]: float(probabilities[0][i]) for i in range(len(label_map))}
    sorted_probs = dict(sorted(class_probs.items(), key=lambda x: x[1], reverse=True))    
    
    st.write(f"**Predicted priority:** {predicted_label}") 
    
    proba_df = pd.DataFrame(sorted_probs.items(), columns=["Priority", "Probability"])
    st.write("**Class probabilities:**")
    st.table(proba_df)
    st.bar_chart(proba_df.set_index("Priority"))
