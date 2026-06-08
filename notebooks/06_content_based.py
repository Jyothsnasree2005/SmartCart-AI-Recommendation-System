import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Articles Dataset...")

articles = pd.read_csv(
    "data/raw/articles.csv/articles_data.csv"
).head(10000)

print("Articles Shape:", articles.shape)

# Fill missing values
articles["prod_name"] = articles["prod_name"].fillna("")
articles["detail_desc"] = articles["detail_desc"].fillna("")
articles["product_type_name"] = articles["product_type_name"].fillna("")
articles["colour_group_name"] = articles["colour_group_name"].fillna("")

print("Creating Product Features...")

articles["features"] = (
    articles["prod_name"] + " " +
    articles["product_type_name"] + " " +
    articles["colour_group_name"] + " " +
    articles["detail_desc"]
)

print("Building TF-IDF Matrix...")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(articles["features"])

print("TF-IDF Shape:", tfidf_matrix.shape)

print("Calculating Similarity Matrix...")

cosine_sim = cosine_similarity(tfidf_matrix)

print("Similarity Matrix Created!")

# Mapping article_id -> index
indices = pd.Series(
    articles.index,
    index=articles["article_id"]
).drop_duplicates()

def get_similar_products(article_id, top_n=5):

    if article_id not in indices:
        print("Article not found")
        return

    idx = indices[article_id]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    product_indices = [i[0] for i in sim_scores]

    return articles[
        ["article_id", "prod_name"]
    ].iloc[product_indices]

# Example Product
sample_article = articles["article_id"].iloc[0]

print("\nSample Product:", sample_article)

recommendations = get_similar_products(sample_article)

print("\nSimilar Products:")
print(recommendations)