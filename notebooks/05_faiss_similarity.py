import pandas as pd
import numpy as np
import faiss
from gensim.models import Word2Vec

print("Loading Item2Vec Model...")

model = Word2Vec.load("models/item2vec.model")

print("Creating Embedding Matrix...")

item_ids = list(model.wv.index_to_key)

vectors = np.array(
    [model.wv[item] for item in item_ids],
    dtype=np.float32
)

print("Items:", len(item_ids))
print("Vector Shape:", vectors.shape)

# Build FAISS index
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(vectors)

print("FAISS Index Created!")
print("Total Indexed Items:", index.ntotal)

# Test Similarity Search
sample_item = item_ids[0]

print("\nSample Item:", sample_item)

query_vector = np.array(
    [model.wv[sample_item]],
    dtype=np.float32
)

distances, indices = index.search(query_vector, 6)

print("\nMost Similar Products:")

for idx in indices[0][1:]:
    print(item_ids[idx])