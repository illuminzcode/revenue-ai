# 🧠 Restaurant Revenue Predictor API

This project provides a simple FastAPI-based REST API to **predict expected revenue** in a restaurant based on the number of people (crowd size). The prediction is powered by a machine learning model trained incrementally using `SGDRegressor` from `scikit-learn`.

---

## 📦 Features

- ✅ Train the model with real revenue data based on crowd size.
- ✅ Predict expected revenue using current crowd.
- ✅ Incremental learning (via `partial_fit`).
- ✅ Model stored using `joblib`.

---

## 🚀 Getting Started

### 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt


1. Train Revenue Model
POST /train_revenue
Train the model with a list of data points (crowd and expected revenue).

Request Body:
[
  { "num_people": 10, "expected_revenue": 2000 },
  { "num_people": 25, "expected_revenue": 4500 }
]
Response:
{
  "message": "Revenue model trained successfully",
  "trained_samples": 2
}

2. Predict Revenue
POST /predict_revenue
Get the expected revenue from the model based on current crowd size.
Request Body:
{ "num_people": 18 }

Response:
{
  "expected_revenue": 3600.25
}

🛠️ Setup Instructions

# 1. Clone the repo
git clone https://github.com/illuminzcode/revenue-ai.git
cd revenue-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn revenue-ai:app --reload