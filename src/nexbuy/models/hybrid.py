"""
Hybrid recommender for the NexBuy recommendation system
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from .base import BaseRecommender
from ..utils.helpers import get_customer_orders_and_products
from ..utils.calculations import get_global_popular_products

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommendation system.
    
    Combines content-based, collaborative filtering, and popularity-based
    recommendations using weighted scoring.
    """
    
    def __init__(self, w_content: float = 0.4, w_collab: float = 0.4, w_pop: float = 0.2):
        """
        Initialize hybrid recommender with weights.
        
        Args:
            w_content: Weight for content-based recommendations
            w_collab: Weight for collaborative filtering recommendations
            w_pop: Weight for popularity-based recommendations
        """
        super().__init__("HybridRecommender")
        
        # Validate weights sum to 1.0
        total_weight = w_content + w_collab + w_pop
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
            
        self.w_content = w_content
        self.w_collab = w_collab
        self.w_pop = w_pop
        
        # Data components
        self.train_data = None
        self.product_indices = None
        self.cosine_sim = None
        self.products = None
        self.product_similarity_df = None
        self.product_popularity = None
    
    def fit(self, train_data: pd.DataFrame, product_indices=None, cosine_sim=None,
            products=None, product_similarity_df=None, product_popularity=None) -> 'HybridRecommender':
        """
        Fit the recommender on training data.
        
        Args:
            train_data (pd.DataFrame): Training transaction data
            product_indices: Product index mapping
            cosine_sim: Cosine similarity matrix
            products: Product information DataFrame
            product_similarity_df: Product similarity matrix DataFrame
            product_popularity: Product popularity DataFrame
            
        Returns:
            self: The fitted recommender
        """
        logger.info("Fitting HybridRecommender...")
        
        self.train_data = train_data
        self.product_indices = product_indices
        self.cosine_sim = cosine_sim
        self.products = products
        self.product_similarity_df = product_similarity_df
        self.product_popularity = product_popularity
        
        self.is_fitted = True
        logger.info(f"Fitted HybridRecommender with weights: content={self.w_content}, collab={self.w_collab}, pop={self.w_pop}")
        return self
    
    def recommend(self, customer_id: Optional[str] = None, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate hybrid recommendations.
        
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

        # --- 1. Calculate Average Content Similarity Scores ---
        purchased_idxs_content = [self.product_indices[pid] for pid in product_ids if pid in self.product_indices]
        if not purchased_idxs_content:
            logger.info(f"No purchased products for customer '{customer_id}' found in content product index.")
            avg_content_sim_scores = np.zeros(len(self.products))
        else:
            # Average similarity to user's purchase history
            valid_idxs = [idx for idx in purchased_idxs_content if idx < self.cosine_sim.shape[0]]
            if not valid_idxs:
                logger.info(f"No valid content-based product indices for customer '{customer_id}'.")
                avg_content_sim_scores = np.zeros(len(self.products))
            else:
                avg_content_sim_scores = sum(self.cosine_sim[idx] for idx in valid_idxs) / len(valid_idxs)

        content_df = pd.DataFrame({
            'Product ID': self.products['Product ID'],
            'content_score': avg_content_sim_scores
        })

        # --- 2. Calculate Average Collaborative Similarity Scores ---
        total_collab_sim = None
        valid_purchased_ids_count = 0
        for pid in product_ids:
            if pid not in self.product_similarity_df.columns:
                continue
            collab_sim = self.product_similarity_df[pid]
            total_collab_sim = collab_sim if total_collab_sim is None else total_collab_sim + collab_sim
            valid_purchased_ids_count += 1

        if total_collab_sim is not None and valid_purchased_ids_count > 0:
            avg_collab_sim = total_collab_sim / valid_purchased_ids_count
            collab_df = pd.DataFrame({
                'Product ID': avg_collab_sim.index,
                'collab_score': avg_collab_sim.values
            })
        else:
            collab_df = pd.DataFrame({
                'Product ID': self.product_similarity_df.columns,
                'collab_score': np.zeros(len(self.product_similarity_df.columns))
            })

        # --- 3. Get Popularity Scores ---
        popularity_df = self.product_popularity[['Product ID', 'popularity_score']].copy()

        # --- 4. Merge All Scores ---
        hybrid_df = popularity_df.merge(content_df, on='Product ID', how='left')
        hybrid_df = hybrid_df.merge(collab_df, on='Product ID', how='left')
        hybrid_df = hybrid_df.fillna(0)

        # --- 5. Calculate Weighted Hybrid Score ---
        hybrid_df['hybrid_score'] = (
            self.w_content * hybrid_df['content_score'] +
            self.w_collab * hybrid_df['collab_score'] +
            self.w_pop * hybrid_df['popularity_score']
        )

        # --- 6. Exclude Already Purchased Products ---
        hybrid_df = hybrid_df[~hybrid_df['Product ID'].isin(product_ids)]

        # --- 7. Sort and Get Top Recommendations ---
        recommendations = hybrid_df.sort_values('hybrid_score', ascending=False).head(top_n)

        # --- 8. Add Product Details ---
        final_recommendations = recommendations.merge(
            self.product_popularity[['Product ID', 'Product Name', 'Category', 'Sub-Category']], 
            on='Product ID', 
            how='left'
        )

        result = final_recommendations[['Product ID', 'Product Name', 'Category', 'Sub-Category', 'hybrid_score']].head(top_n)
        logger.info(f"Generated {len(result)} hybrid recommendations for customer {customer_id}")
        
        return result


__all__ = ['HybridRecommender']
