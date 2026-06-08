import pandas as pd

print("Loading Data...")

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    usecols=["customer_id", "article_id"]
)

articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv",
    usecols=["article_id", "prod_name"]
)

# Popularity Score
popularity = (
    transactions["article_id"]
    .value_counts()
    .reset_index()
)

popularity.columns = ["article_id", "popularity_score"]

top_products = popularity.merge(
    articles,
    on="article_id",
    how="left"
)

print("\nTOP TRENDING PRODUCTS\n")

print(
    top_products[
        ["article_id", "prod_name", "popularity_score"]
    ].head(10)
)

def recommend_customer(customer_id):

    print("\nRecommendations For Customer:")
    print(customer_id)

    return top_products[
        ["article_id", "prod_name"]
    ].head(10)

sample_customer = (
    transactions["customer_id"]
    .iloc[0]
)

recommendations = recommend_customer(sample_customer)

print(recommendations)