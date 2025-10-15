import streamlit as st
import joblib
import numpy as np
import pandas as pd

rf_model = joblib.load("ml/models/pipeline_rf_1.joblib")
log_reg_model = joblib.load("ml/models/pipeline_log_reg_1.joblib")

st.title("Bug Priority Predictor")
text = st.text_area("Paste issue title + description")

if st.button("Predict"):
  pred_class = rf_model.predict([text])[0]
  proba = rf_model.predict_proba([text])
  proba = np.ravel(proba)
  
  class_names = rf_model.classes_
  proba_dict = dict(zip(class_names, proba))
  
  sorted_proba = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
  
  st.write(f"**Predicted priority:** {pred_class}") 
  
  proba_df = pd.DataFrame(sorted_proba.items(), columns=["Priority", "Probability"])
  st.write("**Class probabilities:**")
  st.table(proba_df)
  
  st.bar_chart(proba_df.set_index("Priority"))
    
