import pandas as pd
from scipy.sparse import csr_matrix

print("Loading Transactions...")

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    dtype={
        "article_id": str,
        "customer_id": str
    }
)

print("Original Shape:")
print(transactions.shape)

# Purchase weight = 5
transactions["confidence"] = 5

user_ids = transactions["customer_id"].astype("category")
item_ids = transactions["article_id"].astype("category")

row = user_ids.cat.codes
col = item_ids.cat.codes

confidence_matrix = csr_matrix(
    (
        transactions["confidence"],
        (row, col)
    )
)

print("\nConfidence Matrix Created")

print("Users:", confidence_matrix.shape[0])
print("Items:", confidence_matrix.shape[1])

print("Matrix Shape:")
print(confidence_matrix.shape)