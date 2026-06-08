import pandas as pd
from sqlalchemy import create_engine

# CHANGE PASSWORD BELOW
DATABASE_URL = "postgresql://postgres:Jyothsna%4025@localhost:5432/smartcart_ai"

engine = create_engine(DATABASE_URL)

print("Loading CSV Files...")

customers = pd.read_csv(
    "data/raw/customers.csv/customers_data.csv"
)

articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv",
    dtype={"article_id": str}
)

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    dtype={"article_id": str}
)

print("Uploading Customers...")
customers.to_sql(
    "customers",
    engine,
    if_exists="replace",
    index=False
)

print("Uploading Articles...")
articles[["article_id", "prod_name"]].to_sql(
    "articles",
    engine,
    if_exists="replace",
    index=False
)

print("Reducing Transactions Dataset...")
transactions = transactions.head(500000)

print("Uploading Transactions...")
transactions[["customer_id", "article_id"]].to_sql(
    "transactions",
    engine,
    if_exists="replace",
    index=False,
    chunksize=50000,
    method="multi"
)

print("Upload Complete!")
print("Customers:", len(customers))
print("Articles:", len(articles))
print("Transactions:", len(transactions))