"""
app.py
----------------
Streamlit dashboard for the Intelligent E-Commerce Analytics Suite.
Tabs:
    1. EDA & Sales Trends
    2. Demand Forecasting (Linear Regression)
    3. Customer Segmentation (RFM + K-Means)
    4. Sentiment Analysis (CountVectorizer + Logistic Regression)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_all_data
from ml_models import DemandForecaster, CustomerSegmenter, SentimentAnalyzer

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
# --- CUSTOMIZE HERE: page title, icon, layout ---
st.set_page_config(
    page_title="Intelligent E-Commerce Analytics Suite",
    page_icon="🛒",
    layout="wide",
)


# ----------------------------------------------------------------------
# DATA LOADING (cached so the synthetic data is generated only once)
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_all_data()


customers_df, transactions_df, reviews_df = get_data()

st.title("🛒 Intelligent E-Commerce Analytics Suite")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA & Sales Trends",
    "📈 Demand Forecasting",
    "👥 Customer Segmentation",
    "💬 Sentiment Analysis",
])

# ======================================================================
# TAB 1: EDA & SALES TRENDS
# ======================================================================
with tab1:
    st.subheader("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"₹{transactions_df['total_amount'].sum():,.0f}")
    col2.metric("Total Orders", f"{transactions_df['transaction_id'].nunique():,}")
    col3.metric("Unique Customers", f"{transactions_df['customer_id'].nunique():,}")

    # --- Daily sales trend line chart ---
    daily_sales = (
        transactions_df.groupby(transactions_df["transaction_date"].dt.date)["total_amount"]
        .sum().reset_index()
    )
    fig_trend = px.line(
        daily_sales, x="transaction_date", y="total_amount",
        title="Daily Sales Trend", labels={"transaction_date": "Date", "total_amount": "Revenue (₹)"},
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # --- Category-wise revenue bar chart ---
        cat_sales = transactions_df.groupby("category")["total_amount"].sum().reset_index()
        fig_cat = px.bar(
            cat_sales.sort_values("total_amount", ascending=False),
            x="category", y="total_amount", color="category",
            title="Revenue by Category", labels={"total_amount": "Revenue (₹)"},
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_b:
        # --- Region-wise customer distribution pie chart ---
        region_dist = customers_df["region"].value_counts().reset_index()
        region_dist.columns = ["region", "count"]
        fig_region = px.pie(
            region_dist, names="region", values="count",
            title="Customer Distribution by Region",
        )
        st.plotly_chart(fig_region, use_container_width=True)

    # --- Top 10 best-selling products ---
    top_products = (
        transactions_df.groupby("product_name")["total_amount"]
        .sum().nlargest(10).reset_index()
    )
    fig_top = px.bar(
        top_products, x="total_amount", y="product_name", orientation="h",
        title="Top 10 Products by Revenue", labels={"total_amount": "Revenue (₹)", "product_name": "Product"},
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)


# ======================================================================
# TAB 2: DEMAND FORECASTING (Linear Regression)
# ======================================================================
with tab2:
    st.subheader("Demand Forecasting — Linear Regression")
    st.write("Predicts near-future daily revenue using calendar-based features "
             "(day index / trend, day-of-week, month).")

    col1, col2 = st.columns(2)
    category_choice = col1.selectbox(
        "Select Category", ["All"] + sorted(transactions_df["category"].unique().tolist())
    )
    # --- CUSTOMIZE HERE: forecast horizon range ---
    horizon = col2.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=14)

    if st.button("🚀 Train Forecasting Model", key="train_forecast"):
        forecaster = DemandForecaster()
        metrics, results_df = forecaster.train(transactions_df, category=category_choice)
        future_df = forecaster.forecast_future(horizon_days=horizon)

        st.session_state["forecaster_metrics"] = metrics
        st.session_state["forecaster_results"] = results_df
        st.session_state["forecaster_future"] = future_df
        st.session_state["forecaster_daily"] = forecaster.daily_df

    if "forecaster_metrics" in st.session_state:
        m = st.session_state["forecaster_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.2f}")
        c2.metric("RMSE", f"{m['RMSE']:.2f}")
        c3.metric("R² Score", f"{m['R2']:.3f}")

        # --- Actual vs Predicted (test period) ---
        results_df = st.session_state["forecaster_results"]
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=results_df["date"], y=results_df["sales"], name="Actual"))
        fig_test.add_trace(go.Scatter(x=results_df["date"], y=results_df["predicted_sales"], name="Predicted"))
        fig_test.update_layout(title="Actual vs Predicted Sales (Test Period)",
                                xaxis_title="Date", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig_test, use_container_width=True)

        # --- Historical + Future forecast ---
        daily_df = st.session_state["forecaster_daily"]
        future_df = st.session_state["forecaster_future"]
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["sales"], name="Historical Sales"))
        fig_future.add_trace(go.Scatter(x=future_df["date"], y=future_df["predicted_sales"],
                                         name="Forecasted Sales", line=dict(dash="dash")))
        fig_future.update_layout(title=f"{horizon}-Day Sales Forecast",
                                  xaxis_title="Date", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig_future, use_container_width=True)
    else:
        st.info("Click 'Train Forecasting Model' to generate predictions.")


# ======================================================================
# TAB 3: CUSTOMER SEGMENTATION (RFM + K-Means)
# ======================================================================
with tab3:
    st.subheader("Customer Segmentation — RFM + K-Means Clustering")
    st.write("Segments customers using Recency, Frequency, and Monetary (RFM) metrics.")

    # --- CUSTOMIZE HERE: cluster count range ---
    n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=6, value=4)

    if st.button("🚀 Run Segmentation", key="run_segmentation"):
        segmenter = CustomerSegmenter(n_clusters=n_clusters)
        rfm_df, sil_score = segmenter.fit_predict(transactions_df)
        st.session_state["rfm_df"] = rfm_df
        st.session_state["sil_score"] = sil_score

    if "rfm_df" in st.session_state:
        rfm_df = st.session_state["rfm_df"]
        st.metric("Silhouette Score", f"{st.session_state['sil_score']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            fig_scatter = px.scatter(
                rfm_df, x="Recency", y="Monetary", color="Segment", size="Frequency",
                hover_data=["customer_id"], title="Customer Segments (Recency vs Monetary)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            seg_counts = rfm_df["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig_seg_bar = px.bar(seg_counts, x="Segment", y="Count", color="Segment",
                                  title="Customers per Segment")
            st.plotly_chart(fig_seg_bar, use_container_width=True)

        st.write("#### Segment Profile (Average RFM Values)")
        st.dataframe(
            rfm_df.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2),
            use_container_width=True,
        )

        st.write("#### RFM Table (Sample)")
        st.dataframe(rfm_df.head(20), use_container_width=True)
    else:
        st.info("Click 'Run Segmentation' to cluster customers.")


# ======================================================================
# TAB 4: SENTIMENT ANALYSIS
# ======================================================================
with tab4:
    st.subheader("Sentiment Analysis — CountVectorizer + Logistic Regression")
    st.write("Classifies product reviews into Positive, Neutral, or Negative sentiment.")

    if st.button("🚀 Train Sentiment Model", key="train_sentiment"):
        analyzer = SentimentAnalyzer()
        metrics, cm, class_labels = analyzer.train(reviews_df)
        st.session_state["sentiment_model"] = analyzer
        st.session_state["sentiment_metrics"] = metrics
        st.session_state["sentiment_cm"] = cm
        st.session_state["sentiment_labels"] = class_labels

    if "sentiment_metrics" in st.session_state:
        m = st.session_state["sentiment_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['Accuracy']:.2%}")
        c2.metric("Precision", f"{m['Precision']:.2%}")
        c3.metric("Recall", f"{m['Recall']:.2%}")
        c4.metric("F1 Score", f"{m['F1 Score']:.2%}")

        # --- Confusion matrix heatmap ---
        cm = st.session_state["sentiment_cm"]
        labels = st.session_state["sentiment_labels"]
        fig_cm = px.imshow(
            cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # --- Review sentiment distribution ---
        sent_dist = reviews_df["sentiment_label"].value_counts().reset_index()
        sent_dist.columns = ["Sentiment", "Count"]
        fig_dist = px.pie(sent_dist, names="Sentiment", values="Count", title="Review Sentiment Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.write("#### 🔍 Try Your Own Review")
        user_review = st.text_area("Enter a product review:", "The product quality is amazing and delivery was fast!")
        if st.button("Predict Sentiment"):
            prediction = st.session_state["sentiment_model"].predict(user_review)
            st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.info("Click 'Train Sentiment Model' to classify reviews.")


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "**Intelligent E-Commerce Analytics Suite**\n\n"
        "Demonstrating EDA, Demand Forecasting, "
        "Customer Segmentation, and Sentiment Analysis on synthetic "
        "e-commerce data."
    )
    st.write("---")
    st.write(f"📦 Transactions: {len(transactions_df):,}")
    st.write(f"👥 Customers: {len(customers_df):,}")
    st.write(f"💬 Reviews: {len(reviews_df):,}")