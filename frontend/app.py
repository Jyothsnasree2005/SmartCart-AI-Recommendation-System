import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="SmartCart AI",
    page_icon="🛒",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

st.title("🛒 SmartCart AI")
st.subheader("Personalized E-Commerce Recommendation System")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ALS Model", "✓ Active")

with col2:
    st.metric("Item2Vec", "✓ Active")

with col3:
    st.metric("FAISS Search", "✓ Active")

st.divider()

# ======================
# PROJECT STATISTICS
# ======================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers", "1.37M")

with col2:
    st.metric("Products", "105K")

with col3:
    st.metric("Transactions", "500K")

st.divider()

tab1, tab2 = st.tabs(
    ["🔥 Trending Products", "🎯 Recommendations"]
)

with tab1:

    st.header("Top Trending Products")

    if st.button("Load Trending Products"):

        response = requests.get(
            f"{API_URL}/trending"
        )

        data = response.json()

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True
        )

with tab2:

    st.header("Customer Recommendations")

    customer_id = st.text_input(
        "Enter Customer ID"
    )

    if st.button("Get Recommendations"):

        response = requests.get(
            f"{API_URL}/recommend/{customer_id}"
        )

        data = response.json()

        df = pd.DataFrame(data)

        st.dataframe(
            df,
            use_container_width=True
        )

st.sidebar.title("SmartCart AI")

st.sidebar.info("""
Machine Learning Stack

✓ ALS

✓ Item2Vec

✓ FAISS

✓ Content Based

✓ Hybrid Recommendation
""") 