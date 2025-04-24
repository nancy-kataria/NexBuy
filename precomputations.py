from data_preprocess import superstore_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Product Popularity
product_popularity = superstore_data.groupby('Product ID').agg({
    'Product Name': 'first',
    'Category': 'first',
    'Sub-Category': 'first',
    'Quantity': 'sum',
    'Sales': 'sum'
}).reset_index()

# Normalize popularity score
product_popularity['popularity_score'] = product_popularity['Quantity'] / product_popularity['Quantity'].max()

# 2. Content-Based Info Preparation
superstore_data['product_info'] = (
    superstore_data['Product Name'].astype(str) + ' ' +
    superstore_data['Category'].astype(str) + ' ' +
    superstore_data['Sub-Category'].astype(str)
)

# One row per product
products = superstore_data.drop_duplicates(subset='Product ID')[
    ['Product ID', 'Product Name', 'Category', 'Sub-Category', 'product_info']
]

# 3. TF-IDF Matrix and Cosine Similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(products['product_info'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Product Index Mapping
product_indices = pd.Series(products.index, index=products['Product ID']).drop_duplicates()

# 5. User-Product Matrix and Product Similarity for Collaborative Filtering
# Create user-product interaction matrix
user_product_matrix = superstore_data.pivot_table(
    index='Customer ID',
    columns='Product ID',
    values='Quantity',
    aggfunc='sum'
).fillna(0)


# Compute item-item similarity - similar to item you liked
product_similarity = cosine_similarity(user_product_matrix.T)

# Store as DataFrame
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=user_product_matrix.columns,
    columns=user_product_matrix.columns
)