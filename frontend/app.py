import streamlit as st
import json
import requests
import pandas as pd
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.title("Bug Priority Predictor")
tab1, tab2, tab3 = st.tabs(["Prediction", "History", "Insights"])

# Tab 1: Prediction
with tab1:
    title = st.text_input("Title")
    description = st.text_area("Description")

    if st.button("Predict Priority"):
        if not title or not description:
            st.warning("Please enter both a title and description.")
        else:
            with st.spinner("Predicting..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"title": title, "description": description},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Convert numeric labels to human-readable label
                    predicted_label = data["priority"]
                    confidence = data["confidence"] * 100

                    st.success(f"Predicted Priority: {predicted_label}")

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=confidence,
                            title={
                                "text": f"Priority: {predicted_label} Confidence (%)"
                            },
                            gauge={"axis": {"range": [0, 100]}},
                        )
                    )
                    st.plotly_chart(fig)

                    probs_dict = data.get("probabilities", {})
                    if probs_dict:
                        proba_df = pd.DataFrame(
                            probs_dict.items(), columns=["Priority", "Probability"]
                        )
                        proba_df = proba_df.sort_values("Probability", ascending=False)

                        st.write("### Class Probabilities")
                        st.table(proba_df)
                        st.bar_chart(proba_df.set_index("Priority"))

                else:
                    st.error("Prediction request failed.")


# Tab 2: History
with tab2:
    st.write("### Previous Predictions")

    try:
        response = requests.get(f"{API_URL}/predictions", timeout=60)

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["predictions"])

            if not df.empty:
                for idx, row in df.iterrows():
                    cols = st.columns([4, 1])
                    view_clicked = False

                    with cols[0]:
                        st.write(
                            f"**{row['title']}** | Priority: {row['predicted_label']} | Confidence: {row['confidence']*100:.2f}%"
                        )

                    with cols[1]:
                        if st.button("View", key=f"view_{idx}"):
                            view_clicked = True

                    if view_clicked:
                        st.markdown("---")
                        st.write("**Title:**", row["title"])
                        st.write("**Description:**", row["description"])
                        st.write("**Predicted Label:**", row["predicted_label"])
                        st.write("**Confidence:**", row["confidence"] * 100, "%")
                        st.write("**Timestamp:**", row["timestamp"])
                        st.markdown("---")

            else:
                st.info("No predictions yet.")

        else:
            st.error("Failed to fetch predictions from API.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")


# Tab 3: Insights
with tab3:
    st.write("### Label Distribution")

    try:
        response = requests.get(f"{API_URL}/stats", timeout=60)

        if response.status_code == 200:
            data = response.json()
            counts = pd.DataFrame(data["counts"])
            avg_conf = pd.DataFrame(data["avg_conf"])

            if not counts.empty:
                st.bar_chart(counts.set_index("Priority"))
            else:
                st.info("Not enough data for label counts yet.")

            if not avg_conf.empty:
                st.write("### Average Confidence by Label")
                st.bar_chart(avg_conf.set_index("Priority"))
            else:
                st.info("Not enough data for confidence insights yet.")

        else:
            st.error("Failed to fetch stats from API.")

    except Exception as e:
        st.error(f"Error fetching stats: {e}")
