"""
ml_models.py
----------------
Encapsulates the three ML pipelines used across the dashboard:
  1. DemandForecaster  -> Linear Regression time-series forecasting
  2. CustomerSegmenter -> K-Means clustering on RFM metrics
  3. SentimentAnalyzer -> CountVectorizer + Logistic Regression

Each class exposes train()/fit_predict() and predict() style methods
so that app.py (the Streamlit UI) stays purely presentational.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# ======================================================================
# MODULE 2: DEMAND FORECASTING (Linear Regression)
# ======================================================================
class DemandForecaster:
    """
    Aggregates transaction-level data into a daily sales time series
    and fits a Linear Regression model on calendar features to
    forecast near-future demand (daily revenue).
    """

    def __init__(self):
        self.model = LinearRegression()
        self.daily_df = None
        self.is_trained = False

    def prepare_daily_series(self, transactions_df: pd.DataFrame, category: str = "All") -> pd.DataFrame:
        """Aggregate raw transactions into a continuous daily sales series."""
        df = transactions_df.copy()
        if category != "All":
            df = df[df["category"] == category]

        daily = (
            df.groupby(df["transaction_date"].dt.date)["total_amount"]
            .sum()
            .reset_index()
            .rename(columns={"transaction_date": "date", "total_amount": "sales"})
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)

        # Fill missing calendar days with 0 sales so the series is continuous
        full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(full_range, fill_value=0).rename_axis("date").reset_index()

        # --- Feature engineering ---
        daily["day_index"] = np.arange(len(daily))        # linear trend feature
        daily["day_of_week"] = daily["date"].dt.dayofweek  # 0=Mon ... 6=Sun
        daily["month"] = daily["date"].dt.month

        self.daily_df = daily
        return daily

    def train(self, transactions_df: pd.DataFrame, category: str = "All", test_size: float = 0.2):
        """Train the Linear Regression model on the aggregated daily series."""
        daily = self.prepare_daily_series(transactions_df, category)

        feature_cols = ["day_index", "day_of_week", "month"]
        X = daily[feature_cols]
        y = daily["sales"]

        # --- CUSTOMIZE HERE: test split ratio (chronological split, no shuffle) ---
        split_idx = int(len(daily) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        }

        results_df = daily.iloc[split_idx:].copy()
        results_df["predicted_sales"] = y_pred

        return metrics, results_df

    def forecast_future(self, horizon_days: int = 14) -> pd.DataFrame:
        """Forecast sales for the next `horizon_days` days beyond the training data."""
        if not self.is_trained or self.daily_df is None:
            raise RuntimeError("Model must be trained before forecasting.")

        last_date = self.daily_df["date"].max()
        last_index = self.daily_df["day_index"].max()

        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        future_df = pd.DataFrame({
            "date": future_dates,
            "day_index": np.arange(last_index + 1, last_index + 1 + horizon_days),
            "day_of_week": future_dates.dayofweek,
            "month": future_dates.month,
        })

        future_df["predicted_sales"] = self.model.predict(
            future_df[["day_index", "day_of_week", "month"]]
        ).clip(min=0)  # sales cannot be negative

        return future_df


# ======================================================================
# MODULE 3: CUSTOMER SEGMENTATION (RFM + K-Means)
# ======================================================================
class CustomerSegmenter:
    """
    Computes Recency, Frequency, Monetary (RFM) metrics per customer
    and clusters them using K-Means into behavioural segments.
    """

    def __init__(self, n_clusters: int = 4):
        # --- CUSTOMIZE HERE: number of clusters / KMeans hyperparameters ---
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df = None

    def compute_rfm(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Build the RFM table from raw transactions."""
        snapshot_date = transactions_df["transaction_date"].max() + pd.Timedelta(days=1)

        rfm = transactions_df.groupby("customer_id").agg(
            Recency=("transaction_date", lambda x: (snapshot_date - x.max()).days),
            Frequency=("transaction_id", "count"),
            Monetary=("total_amount", "sum"),
        ).reset_index()

        self.rfm_df = rfm
        return rfm

    def fit_predict(self, transactions_df: pd.DataFrame):
        """Fit K-Means on scaled RFM features and assign cluster + segment labels."""
        rfm = self.compute_rfm(transactions_df)
        features = rfm[["Recency", "Frequency", "Monetary"]]

        scaled_features = self.scaler.fit_transform(features)
        rfm["Cluster"] = self.model.fit_predict(scaled_features)

        sil_score = silhouette_score(scaled_features, rfm["Cluster"])
        rfm["Segment"] = self._label_segments(rfm)

        self.rfm_df = rfm
        return rfm, sil_score

    def _label_segments(self, rfm: pd.DataFrame) -> pd.Series:
        """
        Map numeric cluster IDs to human-readable business labels based on
        the average RFM profile of each cluster. Higher Frequency/Monetary
        and lower Recency indicate a "better" customer segment.
        """
        cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

        cluster_profile["score"] = (
            cluster_profile["Frequency"].rank() +
            cluster_profile["Monetary"].rank() -
            cluster_profile["Recency"].rank()
        )
        ranked_clusters = cluster_profile["score"].sort_values(ascending=False).index.tolist()

        # --- CUSTOMIZE HERE: extend/rename labels if n_clusters changes ---
        label_pool = ["Champions", "Loyal Customers", "At Risk", "Lost Customers",
                      "Potential Loyalists", "New Customers"]
        labels = {cluster: label_pool[i] if i < len(label_pool) else f"Segment {i}"
                  for i, cluster in enumerate(ranked_clusters)}

        return rfm["Cluster"].map(labels)


# ======================================================================
# MODULE 4: SENTIMENT ANALYSIS (CountVectorizer + Logistic Regression)
# ======================================================================
class SentimentAnalyzer:
    """
    A text classification pipeline that converts review text into a
    Bag-of-Words representation (CountVectorizer) and classifies
    sentiment using Logistic Regression.
    """

    def __init__(self, max_features: int = 2000):
        # --- CUSTOMIZE HERE: vectorizer / classifier hyperparameters ---
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.is_trained = False

    def train(self, reviews_df: pd.DataFrame, test_size: float = 0.2):
        """Train the sentiment classifier on labelled review text."""
        X_text = reviews_df["review_text"]
        y = reviews_df["sentiment_label"]

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        cm = confusion_matrix(y_test, y_pred, labels=self.model.classes_)

        return metrics, cm, self.model.classes_

    def predict(self, text: str) -> str:
        """Predict sentiment for a single, user-provided review string."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        vectorized = self.vectorizer.transform([text])
        return self.model.predict(vectorized)[0]