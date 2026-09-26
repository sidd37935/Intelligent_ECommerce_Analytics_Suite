"""
data_loader.py
----------------
Generates synthetic e-commerce data (transactions, customers, reviews)
for the Intelligent E-Commerce Analytics Suite.

All data is generated in-memory using NumPy/Pandas so the project runs
"out of the box" without needing an external dataset, API key, or DB.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- CUSTOMIZE HERE: global reproducibility seed ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# --- CUSTOMIZE HERE: dataset sizing knobs ---
N_CUSTOMERS = 1000
N_TRANSACTIONS = 6000
N_REVIEWS = 2500
DATE_RANGE_DAYS = 365  # 1 year of transaction history

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen",
    "Books", "Beauty & Personal Care", "Sports & Fitness",
]

REGIONS = ["North", "South", "East", "West", "Central"]

PRODUCT_CATALOG = {
    "Electronics": ["Wireless Earbuds", "Smartphone Case", "Bluetooth Speaker", "Power Bank", "Smartwatch"],
    "Fashion": ["Cotton T-Shirt", "Denim Jeans", "Running Shoes", "Winter Jacket", "Leather Wallet"],
    "Home & Kitchen": ["Non-stick Pan", "LED Table Lamp", "Vacuum Flask", "Storage Organizer", "Air Fryer"],
    "Books": ["Fiction Novel", "Self-Help Guide", "Programming Handbook", "Biography", "Cookbook"],
    "Beauty & Personal Care": ["Face Serum", "Shampoo", "Lip Balm", "Sunscreen", "Perfume"],
    "Sports & Fitness": ["Yoga Mat", "Dumbbell Set", "Resistance Bands", "Cycling Helmet", "Sports Bottle"],
}


def generate_customers(n_customers: int = N_CUSTOMERS) -> pd.DataFrame:
    """Generate a synthetic customer master table."""
    customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, n_customers + 1)]
    signup_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, DATE_RANGE_DAYS, size=n_customers), unit="D"
    )

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": np.random.randint(18, 65, size=n_customers),
        "gender": np.random.choice(["Male", "Female", "Other"], size=n_customers, p=[0.48, 0.48, 0.04]),
        "region": np.random.choice(REGIONS, size=n_customers),
        "signup_date": signup_dates,
    })
    return df


def _seasonal_weights(n_days: int) -> np.ndarray:
    """Create a probability weighting over days: slight upward trend + weekend boost."""
    days = np.arange(n_days)
    trend = 1 + (days / n_days) * 0.6  # gradual growth over the year
    weekday = (pd.to_datetime("2023-01-01") + pd.to_timedelta(days, unit="D")).dayofweek
    weekend_boost = np.where(weekday >= 5, 1.4, 1.0)  # Sat/Sun sales boost
    weights = trend * weekend_boost
    return weights / weights.sum()


def generate_transactions(customers_df: pd.DataFrame, n_transactions: int = N_TRANSACTIONS) -> pd.DataFrame:
    """Generate a synthetic transaction log referencing existing customers."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DATE_RANGE_DAYS)

    day_offsets = np.random.choice(range(DATE_RANGE_DAYS), size=n_transactions,
                                    p=_seasonal_weights(DATE_RANGE_DAYS))
    transaction_dates = [start_date + timedelta(days=int(d)) for d in day_offsets]

    categories = np.random.choice(CATEGORIES, size=n_transactions)
    product_names = [np.random.choice(PRODUCT_CATALOG[c]) for c in categories]

    # --- CUSTOMIZE HERE: price ranges could be made category-specific ---
    base_prices = np.random.uniform(199, 4999, size=n_transactions).round(2)
    quantities = np.random.randint(1, 5, size=n_transactions)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{str(i).zfill(6)}" for i in range(1, n_transactions + 1)],
        "customer_id": np.random.choice(customers_df["customer_id"], size=n_transactions),
        "category": categories,
        "product_name": product_names,
        "unit_price": base_prices,
        "quantity": quantities,
        "transaction_date": pd.to_datetime(transaction_dates),
    })
    df["total_amount"] = (df["unit_price"] * df["quantity"]).round(2)
    df = df.sort_values("transaction_date").reset_index(drop=True)
    return df


def generate_reviews(customers_df: pd.DataFrame, n_reviews: int = N_REVIEWS) -> pd.DataFrame:
    """Generate synthetic product reviews with text, star rating, and sentiment label."""

    positive_phrases = [
        "Absolutely love this product, works perfectly!",
        "Great quality and fast delivery, highly recommend.",
        "Exceeded my expectations, will buy again.",
        "Excellent value for money and very durable.",
        "Superb build quality, exactly as described.",
    ]
    neutral_phrases = [
        "Product is okay, nothing special but does the job.",
        "Average quality, meets basic expectations.",
        "It is fine for the price, could be better.",
        "Decent product but packaging could improve.",
        "Works as expected, no major complaints.",
    ]
    negative_phrases = [
        "Very disappointed, product broke within a week.",
        "Poor quality, not worth the money at all.",
        "Delivery was late and item was damaged.",
        "Does not match the description, quite unhappy.",
        "Terrible experience, would not recommend to anyone.",
    ]

    # --- CUSTOMIZE HERE: rating distribution controls sentiment class balance ---
    ratings = np.random.choice([1, 2, 3, 4, 5], size=n_reviews, p=[0.08, 0.10, 0.17, 0.30, 0.35])

    review_texts, sentiments = [], []
    for r in ratings:
        if r >= 4:
            review_texts.append(np.random.choice(positive_phrases))
            sentiments.append("Positive")
        elif r == 3:
            review_texts.append(np.random.choice(neutral_phrases))
            sentiments.append("Neutral")
        else:
            review_texts.append(np.random.choice(negative_phrases))
            sentiments.append("Negative")

    df = pd.DataFrame({
        "review_id": [f"REV{str(i).zfill(6)}" for i in range(1, n_reviews + 1)],
        "customer_id": np.random.choice(customers_df["customer_id"], size=n_reviews),
        "category": np.random.choice(CATEGORIES, size=n_reviews),
        "rating": ratings,
        "review_text": review_texts,
        "sentiment_label": sentiments,
    })
    return df


def load_all_data():
    """Convenience function returning all three synthetic datasets together."""
    customers = generate_customers()
    transactions = generate_transactions(customers)
    reviews = generate_reviews(customers)
    return customers, transactions, reviews


if __name__ == "__main__":
    # Quick smoke test: python data_loader.py
    c, t, r = load_all_data()
    print("Customers:", c.shape)
    print("Transactions:", t.shape)
    print("Reviews:", r.shape)