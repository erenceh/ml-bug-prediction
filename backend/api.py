from fastapi import FastAPI
from huggingface_hub import hf_hub_download
import json
from pydantic import BaseModel
from datetime import datetime
from db import get_connection, init_db
from model_service import predict_priority
import pandas as pd

app = FastAPI()
init_db()


class BugReport(BaseModel):
    title: str
    description: str


LABEL_MAP_PATH = hf_hub_download(
    repo_id="erenceh/ml-bug-priority",
    filename="label_map.json",
    token=st.secrets["HF_TOKEN"],
)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}


@app.post("/predict")
def predict(bug: BugReport):
    try:
        label, confidence, probs = predict_priority(bug.title, bug.description)
        predicted_label_str = label_map[label]
    except Exception as e:
        return {"error": str(e)}

    conn = get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM predictions")
        row_count = cursor.fetchone()[0]

        if row_count >= 10:
            conn.execute(
                "DELETE FROM predictions WHERE id = (SELECT id FROM predictions ORDER BY timestamp ASC LIMIT 1)"
            )

        conn.execute(
            """
      INSERT INTO predictions (title, description, predicted_label, confidence, timestamp) 
      VALUES (?, ?, ?, ?, ?)""",
            (
                bug.title,
                bug.description,
                predicted_label_str,
                confidence,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    except Exception as e:
        return {"error": f"Database insert failed: {e}"}
    finally:
        conn.close()

    # convert probs tensor to dict with readable labels
    probs_dict = {label_map[i]: float(probs[i]) for i in range(len(label_map))}

    return {
        "priority": predicted_label_str,
        "confidence": round(confidence, 3),
        "probabilities": probs_dict,
    }


@app.get("/predictions")
def get_predictions():
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"predictions": rows}


@app.get("/stats")
def get_label_stats():
    conn = get_connection()
    df = pd.read_sql_query("SELECT predicted_label, confidence FROM predictions", conn)
    conn.close()

    if df.empty:
        return {"counts": {}, "avg_conf": {}}

    count_df = df["predicted_label"].value_counts().reset_index()
    count_df.columns = ["Priority", "Count"]

    avg_conf = df.groupby("predicted_label")["confidence"].mean().reset_index()
    avg_conf.columns = ["Priority", "AvgConfidence"]

    return {
        "counts": count_df.to_dict(orient="records"),
        "avg_conf": avg_conf.to_dict(orient="records"),
    }
