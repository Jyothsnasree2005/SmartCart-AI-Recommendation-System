import pandas as pd
import pickle
from gensim.models import Word2Vec

print("Loading Models...")

# ALS
with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)

with open("models/customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

with open("models/article_map.pkl", "rb") as f:
    article_map = pickle.load(f)

# Reverse map
reverse_article_map = {
    v: k for k, v in article_map.items()
}

# Item2Vec
item2vec = Word2Vec.load("models/item2vec.model")

print("Models Loaded!")

# Load transactions
transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    usecols=["customer_id", "article_id"]
)

# Load products
articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv",
    usecols=["article_id", "prod_name"]
)

# Popularity baseline
popularity = (
    transactions["article_id"]
    .value_counts()
)

print("Data Loaded!")

from scipy.sparse import csr_matrix

def recommend_customer(customer_id, n=10):

    if customer_id not in customer_map:
        print("Cold Start User")
        return None

    user_idx = customer_map[customer_id]

    dummy = csr_matrix((1, len(article_map)))

    ids, scores = als_model.recommend(
        user_idx,
        dummy,
        N=n
    )

    recommendations = []

    for item_idx, score in zip(ids, scores):

        article_id = reverse_article_map.get(item_idx)

        if article_id is None:
            continue

        recommendations.append(
            [article_id, score]
        )

    rec_df = pd.DataFrame(
        recommendations,
        columns=["article_id", "als_score"]
    )

    rec_df = rec_df.merge(
        articles,
        on="article_id",
        how="left"
    )

    return rec_df


sample_customer = (
    transactions["customer_id"]
    .iloc[0]
)

print("\nCustomer:")
print(sample_customer)

result = recommend_customer(sample_customer)

print("\nHybrid Recommendations:")
print(result)