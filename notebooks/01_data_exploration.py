import pandas as pd

print("Loading datasets...")

articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv"
)

customers = pd.read_csv(
    "data/raw/customers.csv/customers_data.csv"
)

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv"
)

print("\nDataset Shapes")
print("Articles:", articles.shape)
print("Customers:", customers.shape)
print("Transactions:", transactions.shape)

# ==========================================
# DATA CLEANING
# ==========================================

transactions = transactions.dropna()

print("\nTransactions after cleaning:")
print(transactions.shape)

# ==========================================
# TOP SELLING PRODUCTS
# ==========================================

top_products = (
    transactions["article_id"]
    .value_counts()
    .head(10)
)

print("\nTOP 10 PRODUCTS")
print(top_products)

# ==========================================
# MERGE PRODUCT NAMES
# ==========================================

top_product_details = articles[
    articles["article_id"].isin(top_products.index)
]

print("\nTOP PRODUCT DETAILS")
print(
    top_product_details[
        ["article_id", "prod_name"]
    ].head(10)
)

# ==========================================
# SIMPLE RECOMMENDER FUNCTION
# ==========================================

def recommend_for_customer(customer_id):

    customer_history = transactions[
        transactions["customer_id"] == customer_id
    ]

    if len(customer_history) == 0:
        print("\nNew Customer")
        print("Popular Recommendations:")
        return top_product_details[
            ["article_id", "prod_name"]
        ].head(5)

    purchased_items = set(
        customer_history["article_id"]
    )

    recommendations = articles[
        ~articles["article_id"].isin(
            purchased_items
        )
    ]

    return recommendations[
        ["article_id", "prod_name"]
    ].head(5)

# ==========================================
# TEST RECOMMENDATION
# ==========================================

sample_customer = (
    transactions["customer_id"]
    .iloc[0]
)

print("\nSample Customer:")
print(sample_customer)

recommendations = recommend_for_customer(
    sample_customer
)

print("\nRecommended Products")
print(recommendations)