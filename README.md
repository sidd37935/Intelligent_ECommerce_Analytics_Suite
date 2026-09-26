# 🛒 Intelligent E-Commerce Analytics Suite

## Abstract
The **Intelligent E-Commerce Analytics Suite** is an end-to-end analytics and
machine learning dashboard that simulates a real-world e-commerce data
pipeline. It ingests synthetic transaction, customer, and review data and
applies four analytical modules — Exploratory Data Analysis, Demand
Forecasting, Customer Segmentation, and Sentiment Analysis — to surface
actionable business insights. The system is built entirely on an accessible,
open-source Python stack (Pandas, NumPy, Scikit-Learn, Plotly, Streamlit) and
is designed to be runnable out-of-the-box without any external data
dependency, making it well suited for an academic mini-project demonstration.

## System Architecture
                     ┌────────────────────────┐
                     │      data_loader.py     │
                     │  (Synthetic Data Engine)│
                     │  - Customers            │
                     │  - Transactions         │
                     │  - Reviews              │
                     └────────────┬─────────────┘
                                  │  Pandas DataFrames
                                  ▼
                     ┌────────────────────────┐
                     │       ml_models.py       │
                     │   (ML Pipeline Layer)    │
                     │ ┌──────────────────────┐ │
                     │ │ DemandForecaster      │ │  Linear Regression
                     │ │ CustomerSegmenter     │ │  K-Means (RFM)
                     │ │ SentimentAnalyzer     │ │  CountVec + LogReg
                     │ └──────────────────────┘ │
                     └────────────┬─────────────┘
                                  │  Metrics + Predictions
                                  ▼
                     ┌────────────────────────┐
                     │         app.py           │
                     │  (Streamlit UI Layer)    │
                     │  Tab1: EDA               │
                     │  Tab2: Forecasting       │
                     │  Tab3: Segmentation      │
                     │  Tab4: Sentiment         │
                     └────────────┬─────────────┘
                                  │  Plotly Charts
                                  ▼
                          ┌───────────────┐
                          │   End User /   │
                          │   Browser UI   │
                          └───────────────┘


## Setup & Installation Guide

### 1. Clone / Download the Project
Place `app.py`, `data_loader.py`, `ml_models.py`, and `requirements.txt` in a
single project folder.

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The dashboard will automatically open at `http://localhost:8501`.

## Datasets Used
All data is **synthetically generated at runtime** — no external files
required.

| Dataset      | Records (default) | Key Fields                                                        |
|--------------|--------------------|---------------------------------------------------------------------|
| Customers    | 1,000              | customer_id, age, gender, region, signup_date                       |
| Transactions | 6,000              | transaction_id, customer_id, category, product_name, total_amount, transaction_date |
| Reviews      | 2,500              | review_id, customer_id, category, rating, review_text, sentiment_label |

## Algorithms & Models

| Module                  | Algorithm                          | Library      | Purpose                                    |
|--------------------------|-------------------------------------|--------------|---------------------------------------------|
| EDA & Sales Trends        | Aggregation & Visualization         | Pandas/Plotly| Descriptive analytics                       |
| Demand Forecasting        | Linear Regression                   | Scikit-Learn | Predict near-future daily revenue           |
| Customer Segmentation     | K-Means Clustering on RFM           | Scikit-Learn | Group customers into behavioural segments   |
| Sentiment Analysis        | CountVectorizer + Logistic Regression | Scikit-Learn | Classify review sentiment (Pos/Neu/Neg)     |

## Project Structure
ecommerce-analytics-suite/
│
├── app.py # Streamlit dashboard (UI layer)
├── data_loader.py # Synthetic data generation
├── ml_models.py # ML pipelines (train/predict/evaluate)
├── requirements.txt # Dependencies
├── README.md # This file
└── PROJECT_REPORT.md # Formal academic report

## Future Enhancements
- Replace synthetic data with real e-commerce datasets (e.g., Kaggle Olist).
- Upgrade forecasting to ARIMA/Prophet/LSTM for seasonality handling.
- Use TF-IDF or transformer embeddings (BERT) for sentiment analysis.
- Add a recommendation engine module (collaborative filtering).
- Persist trained models using `joblib` for reuse across sessions.
