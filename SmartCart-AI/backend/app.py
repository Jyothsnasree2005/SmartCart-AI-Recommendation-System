from sqlalchemy import create_engine
from fastapi import FastAPI
import pandas as pd
import pickle
from scipy.sparse import coo_matrix

app = FastAPI(title="SmartCart Recommendation API")

DATABASE_URL = "postgresql://postgres:Jyothsna%4025@localhost:5432/smartcart_ai"

engine = create_engine(DATABASE_URL)

print("Loading Models...")

# ==========================
# LOAD MODELS
# ==========================

with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

with open("models/customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

with open("models/article_map.pkl", "rb") as f:
    article_map = pickle.load(f)
    
    print("Article Map Sample:")
print(list(article_map.keys())[:10])

with open("models/customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

print("Customer Map Sample:")
print(list(customer_map.keys())[:5])

print("Customer Key Type:")
print(type(list(customer_map.keys())[0]))

# ==========================
# LOAD DATA
# ==========================

articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv",
    dtype={"article_id": str}
)

articles["article_id"] = (
    articles["article_id"]
    .astype(str)
    .str.lstrip("0")
)

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    dtype={
        "customer_id": str,
        "article_id": str
    },
    nrows=500000
)

print("Transaction Article Type:")
print(type(transactions["article_id"].iloc[0]))

transactions["article_id"] = (
    transactions["article_id"]
    .astype(str)
    .str.lstrip("0")
)
print("Transaction Article Sample:")
print(transactions["article_id"].head())
print("Transaction Customer Sample:")
print(transactions["customer_id"].head())

# ==========================
# BUILD CUSTOMER-ITEM MATRIX
# ==========================

# Convert article_map keys from int -> string
article_map = {
    str(int(k)): v
    for k, v in article_map.items()
}

# Normalize article ids
transactions["article_id"] = (
    transactions["article_id"]
    .astype(str)
    .str.lstrip("0")
)

articles["article_id"] = (
    articles["article_id"]
    .astype(str)
    .str.lstrip("0")
)

print("Original:", len(transactions))

# Customer filter
transactions1 = transactions[
    transactions["customer_id"].isin(customer_map.keys())
]

print("After customer filter:", len(transactions1))

# Article filter
transactions2 = transactions1[
    transactions1["article_id"].isin(article_map.keys())
]

print("After article filter:", len(transactions2))

# Use filtered data only
transactions = transactions2.copy()

transactions["customer_idx"] = (
    transactions["customer_id"]
    .map(customer_map)
)

transactions["article_idx"] = (
    transactions["article_id"]
    .map(article_map)
)

transactions = transactions.dropna()

print("After dropna:", len(transactions))

if len(transactions) > 0:

    customer_item_matrix = coo_matrix(
        (
            [1] * len(transactions),
            (
                transactions["customer_idx"].astype(int),
                transactions["article_idx"].astype(int)
            )
        )
    ).tocsr()

else:

    print("WARNING: No matching transactions found")
    customer_item_matrix = None

print("Models Loaded Successfully!")

# ==========================
# HOME
# ==========================

@app.get("/")
def home():
    return {
        "project": "SmartCart AI",
        "status": "Running",
        "model": "ALS + Item2Vec + Hybrid"
    }

# ==========================
# TRENDING PRODUCTS
# ==========================

@app.get("/trending")
def trending(n: int = 10):

    query = f"""
    SELECT
        t.article_id,
        COUNT(*) as count,
        a.prod_name
    FROM transactions t
    JOIN articles a
        ON t.article_id = a.article_id
    GROUP BY t.article_id, a.prod_name
    ORDER BY count DESC
    LIMIT {n}
    """

    result = pd.read_sql(query, engine)

    return result.to_dict(orient="records")
# ==========================
# SIMILAR PRODUCT
# ==========================

@app.get("/similar/{item_id}")
def similar(item_id: str):

    item_id = item_id.lstrip("0")

    articles["article_id_clean"] = (
        articles["article_id"]
        .astype(str)
        .str.lstrip("0")
    )

    product = articles[
        articles["article_id_clean"] == item_id
    ][["article_id", "prod_name"]]
    
    if len(product) == 0:
        return {"message": "Product Not Found"}

    return product.to_dict(orient="records")

# ==========================
# ALS RECOMMENDATIONS
# ==========================

@app.get("/recommend/{customer_id}")
def recommend(customer_id: str):

    if customer_id not in customer_map:
        return {"message": "Customer Not Found"}

    try:

        customer_idx = customer_map[customer_id]

        ids, scores = als_model.recommend(
            customer_idx,
            customer_item_matrix[customer_idx],
            N=10
        )

        results = []

        for article_idx, score in zip(ids, scores):

            article_row = articles.iloc[int(article_idx)]

            results.append({
                "article_id": str(article_row["article_id"]),
                "product_name": str(article_row["prod_name"]),
                "score": float(score)
            })

        return results

    except Exception as e:

        return {
            "error": str(e)
        }