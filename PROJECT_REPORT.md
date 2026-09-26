# INTELLIGENT E-COMMERCE ANALYTICS SUITE
### A Machine Learning Based Mini-Project Report

**Submitted by:** [Your Full Name]
**Roll No.:** [Your Roll Number]
**Department:** Artificial Intelligence & Machine Learning
**Program:** B.Tech (AI/ML)
**Academic Year:** [e.g., 2025–2026]
**Guided by:** [Faculty Mentor Name]

---

## Chapter 1: Introduction & Problem Statement

### 1.1 Introduction
E-commerce platforms generate massive volumes of transactional, behavioural,
and textual data every day. Extracting meaningful patterns from this data —
such as sales trends, future demand, customer behaviour, and product
perception — is critical for data-driven business decision-making. This
project, the **Intelligent E-Commerce Analytics Suite**, implements a
unified analytics dashboard that combines classical machine learning
techniques with interactive visual analytics to address these needs.

### 1.2 Problem Statement
Small and mid-sized e-commerce businesses often lack accessible tools to:
1. Visualize sales performance and trends across products/categories.
2. Forecast short-term product demand for inventory planning.
3. Segment customers to enable targeted marketing.
4. Automatically gauge customer sentiment from textual reviews.

This project addresses these four gaps through a single, lightweight,
open-source dashboard built on Python's data science ecosystem.

### 1.3 Objectives
- To design a synthetic e-commerce data generator simulating real-world
  transaction, customer, and review data.
- To implement supervised (Linear Regression, Logistic Regression) and
  unsupervised (K-Means) machine learning models.
- To build an interactive Streamlit dashboard with Plotly visualizations
  for each analytical module.
- To evaluate each model using appropriate quantitative metrics.

### 1.4 Scope
The project scope is limited to demonstrating the ML pipeline and dashboard
on synthetic data; however, the modular design allows the data source to be
swapped for real-world data with minimal code changes.

---

## Chapter 2: Literature Survey / System Requirements

### 2.1 Literature Survey (Summary)
Prior work in e-commerce analytics commonly separates concerns into (a)
descriptive analytics (dashboards), (b) predictive analytics (forecasting
models such as ARIMA, regression, or deep learning), (c) customer
analytics (RFM-based segmentation, widely used in CRM systems), and (d)
text analytics (sentiment classification using Bag-of-Words, TF-IDF, or
transformer-based embeddings). This project draws on well-established,
interpretable techniques — Linear Regression, K-Means, and Logistic
Regression — that are appropriate for an academic mini-project while
remaining production-relevant in industry.

### 2.2 System Requirements

**Hardware Requirements**
| Component | Minimum Specification |
|-----------|------------------------|
| Processor | Intel i3 / equivalent   |
| RAM       | 4 GB (8 GB recommended) |
| Storage   | 500 MB free space       |
| Display   | 1366×768 or higher      |

**Software Requirements**
| Component        | Specification              |
|-------------------|-----------------------------|
| Operating System   | Windows / macOS / Linux    |
| Programming Language | Python 3.9+              |
| IDE                | VS Code / PyCharm / Jupyter |
| Libraries          | Pandas, NumPy, Scikit-Learn, Plotly, Streamlit |
| Browser            | Chrome / Edge / Firefox (for Streamlit UI) |

---

## Chapter 3: System Architecture & Data Flow

### 3.1 Architecture Overview
The system follows a three-layer modular architecture:

1. **Data Layer** (`data_loader.py`) — generates synthetic customers,
   transactions, and reviews as Pandas DataFrames.
2. **Model Layer** (`ml_models.py`) — encapsulates all ML logic as
   independent, reusable classes (`DemandForecaster`, `CustomerSegmenter`,
   `SentimentAnalyzer`).
3. **Presentation Layer** (`app.py`) — a Streamlit application that
   orchestrates data flow between the data and model layers and renders
   interactive Plotly visualizations across four tabs.

### 3.2 Data Flow Diagram
[Synthetic Data Generation]
│
▼
[Raw DataFrames: customers, transactions, reviews]
│
├──► [EDA Module] ──► Aggregations ──► Plotly Charts
│
├──► [Forecasting Module] ──► Feature Engineering ──► Linear Regression
│ └──► Train/Test Split ──► Metrics (MAE, RMSE, R²) ──► Forecast
│
├──► [Segmentation Module] ──► RFM Computation ──► StandardScaler
│ └──► K-Means ──► Cluster Labels ──► Segment Naming
│
└──► [Sentiment Module] ──► CountVectorizer ──► Logistic Regression
└──► Train/Test Split ──► Metrics (Accuracy, Precision, Recall, F1)

### 3.3 Module Interaction
`app.py` calls `data_loader.load_all_data()` once (cached), then passes the
resulting DataFrames into the respective `ml_models.py` classes on user
interaction (button clicks), keeping computation on-demand and the UI
responsive.

---

## Chapter 4: Methodology & Algorithm Details

### 4.1 Demand Forecasting — Linear Regression
Daily transactions are aggregated into a continuous time series of total
revenue. Three calendar-derived features are engineered:
- `day_index`: a monotonically increasing integer capturing overall trend.
- `day_of_week`: captures weekly seasonality (e.g., weekend spikes).
- `month`: captures broader seasonal/monthly effects.

A **Linear Regression** model is trained on a chronological train/test split
(80/20) to predict `total_amount` (daily sales). The fitted model is then
extrapolated forward to forecast a user-defined horizon (7–60 days).

**Model equation:**
`sales ≈ β₀ + β₁(day_index) + β₂(day_of_week) + β₃(month)`

### 4.2 Customer Segmentation — RFM + K-Means
For each customer, three metrics are computed:
- **Recency (R):** Days since the customer's last purchase.
- **Frequency (F):** Total number of transactions.
- **Monetary (M):** Total amount spent.

Features are standardized using `StandardScaler` (zero mean, unit variance)
before applying **K-Means clustering** (default k=4). Clusters are then
ranked by a composite score (high F/M, low R) and mapped to interpretable
business labels: *Champions, Loyal Customers, Potential Loyalists, At Risk,*
etc.

### 4.3 Sentiment Analysis — CountVectorizer + Logistic Regression
Review text is converted into a **Bag-of-Words** representation using
`CountVectorizer` (with English stop-word removal and a max vocabulary size
of 2,000 features). A **multinomial Logistic Regression** classifier is
trained on an 80/20 stratified train/test split to predict one of three
classes: *Positive, Neutral, Negative*.

---

## Chapter 5: Results & Visual Performance Metrics

### 5.1 Demand Forecasting Results
| Metric | Description | Typical Range (synthetic data) |
|--------|-------------|----------------------------------|
| MAE    | Mean Absolute Error in revenue prediction | Reported live in dashboard |
| RMSE   | Root Mean Squared Error | Reported live in dashboard |
| R²     | Proportion of variance explained | Reported live in dashboard |

*(Exact values depend on the randomly generated dataset seed and selected
category; re-run the "Train Forecasting Model" button to view current
metrics.)*

### 5.2 Customer Segmentation Results
| Metric            | Description                                      |
|--------------------|---------------------------------------------------|
| Silhouette Score   | Measures cluster cohesion/separation (−1 to 1); higher is better |
| Segment Sizes      | Distribution of customers across identified segments |

### 5.3 Sentiment Analysis Results
| Metric     | Description                                     |
|------------|--------------------------------------------------|
| Accuracy   | Overall percentage of correctly classified reviews |
| Precision  | Weighted precision across Positive/Neutral/Negative |
| Recall     | Weighted recall across all classes               |
| F1 Score   | Harmonic mean of precision and recall            |
| Confusion Matrix | Visualized as a Plotly heatmap in Tab 4    |

> **Note:** All metrics above are computed live within the Streamlit
> application each time a model is trained, and are displayed as metric
> cards and Plotly charts in the respective dashboard tabs. Screenshots of
> these visual outputs should be inserted here in the final submitted
> report.

---

## Chapter 6: Conclusion & Future Scope

### 6.1 Conclusion
The Intelligent E-Commerce Analytics Suite successfully demonstrates how
classical, interpretable machine learning algorithms — Linear Regression,
K-Means Clustering, and Logistic Regression — can be integrated into a
single, cohesive analytics dashboard to address real business questions:
sales visibility, demand planning, customer understanding, and review
sentiment. The modular codebase (separating data generation, ML logic, and
UI) ensures the project is maintainable, extensible, and suitable as a
foundation for more advanced future work.

### 6.2 Future Scope
- **Advanced Forecasting:** Incorporate ARIMA, Facebook Prophet, or LSTM
  networks to better capture seasonality and non-linear trends.
- **Deep Learning for Sentiment:** Replace Bag-of-Words with TF-IDF or
  transformer-based embeddings (e.g., DistilBERT) for improved accuracy.
- **Real-World Data Integration:** Connect to live e-commerce APIs or
  public datasets (e.g., Olist, Amazon Reviews) for real deployment.
- **Recommendation System:** Add a collaborative-filtering or
  content-based recommender as a fifth module.
- **Model Persistence & MLOps:** Save trained models with `joblib`/`pickle`
  and add a lightweight retraining pipeline.
- **Authentication & Multi-user Support:** Extend the Streamlit app for
  role-based access (admin vs. analyst views).

---

## References
1. Scikit-learn Documentation — https://scikit-learn.org/stable/
2. Streamlit Documentation — https://docs.streamlit.io/
3. Plotly Python Graphing Library — https://plotly.com/python/
4. Pandas Documentation — https://pandas.pydata.org/docs/
5. Fader, P. & Hardie, B. — "RFM and CLV: Using Iso-value Curves for
   Customer Base Analysis" (customer segmentation methodology reference)