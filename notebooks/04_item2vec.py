import pandas as pd
from gensim.models import Word2Vec

print("Loading transactions...")

transactions = pd.read_csv(
    "data/raw/transactions_train.csv/transactions_data.csv",
    usecols=["customer_id", "article_id"]
)

# Use subset for faster training
transactions = transactions.head(500000)

print("Creating purchase sequences...")

transactions["article_id"] = transactions["article_id"].astype(str)

user_sequences = (
    transactions.groupby("customer_id")["article_id"]
    .apply(list)
    .tolist()
)

print("Number of sequences:", len(user_sequences))

print("Training Item2Vec Model...")

model = Word2Vec(
    sentences=user_sequences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)

print("Training Complete!")

model.save("models/item2vec.model")

print("Model Saved!")

# Example product
sample_item = user_sequences[0][0]

print("\nSample Item:", sample_item)

print("\nMost Similar Products:")

similar = model.wv.most_similar(sample_item, topn=10)

for item, score in similar:
    print(item, round(score, 4))