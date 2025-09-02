"""
Core calculation functions for the NexBuy recommendation system
"""
import pandas as pd
from typing import Optional


def get_global_popular_products(product_popularity: pd.DataFrame, top_n: int = 5, by: str = 'Quantity') -> pd.DataFrame:
    """
    Recommends top-N globally popular products. Sorts all products based on a specified 
    metric ('Quantity' or 'Sales') and returns the top N. Does not consider customer history.

    Args:
        product_popularity (pd.DataFrame): DataFrame containing product popularity data
        top_n (int, optional): The number of products to recommend. Defaults to 5.
        by (str, optional): The metric to sort popularity by ('Quantity' or 'Sales'). Defaults to 'Quantity'.

    Returns:
        pd.DataFrame: A DataFrame containing the top N popular products with columns 
                      ['Product ID', 'Product Name', 'Category', 'Sub-Category', <by>]. 
                      Returns an empty DataFrame if an invalid 'by' parameter is provided.

    Raises:
        ValueError: If 'by' is not 'Quantity' or 'Sales'.
    """
    if by not in ['Quantity', 'Sales']:
        raise ValueError("Parameter 'by' must be either 'Quantity' or 'Sales'")

    return product_popularity.sort_values(by=by, ascending=False).head(top_n)[
        ['Product ID', 'Product Name', 'Category', 'Sub-Category', by]
    ]


def get_content_similar_items(product_id: str, product_indices: pd.Series, 
                             cosine_sim, products: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommends products similar to a given product based on content.

    Uses precomputed TF-IDF vectors and cosine similarity based on product
    name, category, and sub-category.

    Args:
        product_id (str): The ID of the product to find similar items for.
        product_indices (pd.Series): Series mapping product IDs to indices
        cosine_sim: Cosine similarity matrix
        products (pd.DataFrame): DataFrame containing product information
        top_n (int, optional): The number of similar products to return. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the top_n similar products with
                      columns ['Product ID', 'Product Name', 'Category', 'Sub-Category'].
                      Returns an empty DataFrame if the product_id is not found.
    """
    if product_id not in product_indices.index:
        return pd.DataFrame(columns=['Product ID', 'Product Name', 'Category', 'Sub-Category'])

    idx = product_indices[product_id]

    if idx >= cosine_sim.shape[0]:  # safety check
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_idxs = [i[0] for i in sim_scores]

    return products.iloc[product_idxs][['Product ID', 'Product Name', 'Category', 'Sub-Category']]


def get_collaborative_similar_items(product_id: str, product_similarity_df: pd.DataFrame,
                                  product_popularity: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Recommends products similar to a given product using item-item collaborative filtering.

    Uses a precomputed product similarity matrix based on user co-purchase patterns.

    Args:
        product_id (str): The ID of the product to find collaboratively similar items for.
        product_similarity_df (pd.DataFrame): Product similarity matrix
        product_popularity (pd.DataFrame): Product popularity data for merging
        top_n (int, optional): The number of similar products to return. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the top_n similar products
                      with columns ['Product ID', 'Similarity Score', 'Product Name',
                      'Category', 'Sub-Category']. Returns an empty DataFrame if the
                      product_id is not found in the similarity matrix.
    """
    if product_id not in product_similarity_df.columns:
        print(f"Product {product_id} not found in dataset.")
        return pd.DataFrame()
    
    similar_scores = product_similarity_df[product_id].sort_values(ascending=False)
    recommended = similar_scores[1:top_n+1].reset_index()
    recommended.columns = ['Product ID', 'Similarity Score']
    
    return recommended.merge(
        product_popularity[['Product ID', 'Product Name', 'Category', 'Sub-Category']],
        on='Product ID', how='left'
    )


__all__ = [
    'get_global_popular_products',
    'get_content_similar_items',
    'get_collaborative_similar_items'
]
