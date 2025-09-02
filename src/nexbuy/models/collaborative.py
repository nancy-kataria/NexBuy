"""
Collaborative filtering recommender for the NexBuy recommendation system
"""
import pandas as pd
from typing import Optional
import logging

from .base import BaseRecommender
from ..utils.helpers import get_customer_orders_and_products
from ..utils.calculations import get_global_popular_products

logger = logging.getLogger(__name__)


class CollaborativeRecommender(BaseRecommender):
    """
    Collaborative filtering recommendation system.
    
    Recommends products based on item-item collaborative filtering using
    user co-purchase patterns to find similar items.
    """
    
    def __init__(self):
        super().__init__("CollaborativeRecommender")
        self.train_data = None
        self.product_similarity_df = None
        self.product_popularity = None
    
    def fit(self, train_data: pd.DataFrame, product_similarity_df=None, 
            product_popularity=None) -> 'CollaborativeRecommender':
        """
        Fit the recommender on training data.
        
        Args:
            train_data (pd.DataFrame): Training transaction data
            product_similarity_df: Product similarity matrix DataFrame
            product_popularity: Product popularity DataFrame
            
        Returns:
            self: The fitted recommender
        """
        logger.info("Fitting CollaborativeRecommender...")
        
        self.train_data = train_data
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity
        
        self.is_fitted = True
        logger.info("Fitted CollaborativeRecommender")
        return self
    
    def recommend(self, customer_id: Optional[str] = None, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate collaborative filtering recommendations.
        
        Args:
            customer_id (str, optional): Customer ID to generate recommendations for
            top_n (int): Number of recommendations to generate
            **kwargs: Additional parameters (currently unused)
            
        Returns:
            pd.DataFrame: Recommended products
        """
        self._check_fitted()
        self._validate_input(customer_id, top_n)
        
        # Case 1: No customer → return top global products
        if customer_id is None:
            return get_global_popular_products(self.product_popularity, top_n=top_n)

        order_history, product_ids = get_customer_orders_and_products(customer_id, self.train_data)
        if order_history.empty:
            return pd.DataFrame()

        # If user has multiple purchases, accumulate similarity
        total_collab_scores = None
        valid_count = 0
        for pid in product_ids:
            if pid not in self.product_similarity_df.columns:
                continue
            product_scores = self.product_similarity_df[pid]
            total_collab_scores = product_scores if total_collab_scores is None else total_collab_scores + product_scores
            valid_count += 1

        if total_collab_scores is None or valid_count == 0:
            logger.info(f"No valid products found for similarity for customer '{customer_id}'.")
            return pd.DataFrame()

        # Normalize if multiple products
        total_collab_scores = total_collab_scores / valid_count

        # Remove already purchased products
        total_collab_scores = total_collab_scores.drop(labels=product_ids, errors='ignore')

        # Get top similar product IDs
        top_scores = total_collab_scores.sort_values(ascending=False)
        top_ids = top_scores.head(top_n).index.tolist()

        # Fetch recommended products
        recommendations = self.product_popularity[self.product_popularity['Product ID'].isin(top_ids)].copy()
        recommendations['Similarity Score'] = top_scores[top_ids].values

        # Fallback if not enough items
        if len(recommendations) < top_n:
            logger.info(f"Only {len(recommendations)} collaborative recommendations found. Adding fallback items.")
            fallback = get_global_popular_products(self.product_popularity, top_n=top_n * 2)
            fallback = fallback[~fallback['Product ID'].isin(product_ids + top_ids)]
            fallback = fallback.head(top_n - len(recommendations))
            fallback['Similarity Score'] = 0
            recommendations = pd.concat([recommendations, fallback])

        result = recommendations.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category', 'Similarity Score']]
        logger.info(f"Generated {len(result)} collaborative recommendations for customer {customer_id}")
        
        return result


__all__ = ['CollaborativeRecommender']
