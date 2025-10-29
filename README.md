# ML Bug Prediction

This project is a **Bug Priority Prediction System** that uses a **BERT model** to predict the priority level of software bug issues based on their **Title** and **Description**.

The model was trained on the **ISEC Data Challenge Dataset** and integrates with a **Streamlit front-end** and **FastAPI back-end** for real-time predictions. It demonstrates how natural processing (NLP) can assist software teams in triaging and managing issue reports efficiently.

**ISEC Data Challenge Dataset**: [https://www.kaggle.com/competitions/isec-sdc-2025/overview](https://www.kaggle.com/competitions/isec-sdc-2025/overview)

---

## Features

- Predicts bug priority levels using a fine-tuned **BERT model**
- Interactive web interface built with **Streamlit**
- RESTful API built with **FastAPI**
- Presistent local storage of predictions via **SQLite**
- Data visualization using **Plotly**

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/erenceh/ml-bug-prediction.git
cd ml-bug-prediction
```

### 2. Create and Activate the Conda Environment

```cli
conda env create -f environment.yml
conda activate bug-prediction-env
```

## Running the Project Locally

### 1. Start the FastAPI Backend

Make folder named `data/` in the root directory if it doesn't exist:

```cli
mkdir data
```

This folder is required for the SQLite database (`predictions.db`) used by the backend to store prediction history.

In the project root folder, run:

```cli

uvicorn backend.api:app --reload
```

### 2. Start the Streamlis Frontend

In a new terminal (still in the root folder), run:

```cli
conda activate bug-prediction-env
strealit run frontend/app.py
```

## Example Usage

1. Enter a bug title and description in the Streamlit web app.
2. Click Predict Priority.
3. View:
   - The predicted priority label
   - Model confidence gauge
   - Class probabilities chart
   - History of past predictions

## License

This project is provided for educational and demonstration purposes under the MIT License.

## Credits

Developed by Eric Rencehausen
Trained using the ISEC Data Challenge Dataset and fine-tuned with Hugging Face Transformers.
