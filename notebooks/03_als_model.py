import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

print("Loading Data...")

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    usecols=["customer_id", "article_id"]
)

transactions = transactions.head(500000)

# Encode IDs
customer_map = {
    id_: idx
    for idx, id_ in enumerate(
        transactions["customer_id"].unique()
    )
}

article_map = {
    id_: idx
    for idx, id_ in enumerate(
        transactions["article_id"].unique()
    )
}

transactions["user_idx"] = transactions["customer_id"].map(customer_map)
transactions["item_idx"] = transactions["article_id"].map(article_map)

# Sparse Matrix
matrix = coo_matrix(
    (
        np.ones(len(transactions)),
        (
            transactions["item_idx"],
            transactions["user_idx"]
        )
    )
)

print("Training ALS Model...")

model = AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=20
)

model.fit(matrix)

print("ALS Training Complete!")

# Save maps
import pickle

with open("models/customer_map.pkl", "wb") as f:
    pickle.dump(customer_map, f)

with open("models/article_map.pkl", "wb") as f:
    pickle.dump(article_map, f)

with open("models/als_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved!")