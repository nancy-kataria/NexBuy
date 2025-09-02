"""
Content-based recommender for the NexBuy recommendation system
"""
import pandas as pd
from typing import Optional
import logging

from .base import BaseRecommender
from ..utils.helpers import get_customer_orders_and_products
from ..utils.calculations import get_content_similar_items, get_global_popular_products

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommendation system.
    
    Recommends products similar to items the customer has purchased,
    using TF-IDF and cosine similarity on product features.
    """
    
    def __init__(self):
        super().__init__("ContentBasedRecommender")
        self.train_data = None
        self.product_indices = None
        self.cosine_sim = None
        self.products = None
        self.product_popularity = None
    
    def fit(self, train_data: pd.DataFrame, product_indices=None, cosine_sim=None, 
            products=None, product_popularity=None) -> 'ContentBasedRecommender':
        """
        Fit the recommender on training data.
        
        Args:
            train_data (pd.DataFrame): Training transaction data
            product_indices: Product index mapping
            cosine_sim: Cosine similarity matrix
            products: Product information DataFrame
            product_popularity: Product popularity DataFrame
            
        Returns:
            self: The fitted recommender
        """
        logger.info("Fitting ContentBasedRecommender...")
        
        self.train_data = train_data
        self.product_indices = product_indices
        self.cosine_sim = cosine_sim
        self.products = products
        self.product_popularity = product_popularity
        
        self.is_fitted = True
        logger.info("Fitted ContentBasedRecommender")
        return self
    
    def recommend(self, customer_id: Optional[str] = None, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate content-based recommendations.
        
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

        # Get last product bought
        last_purchase = order_history.sort_values('Order Date', ascending=False).iloc[0]
        last_product_id = last_purchase['Product ID']
        last_product_name = last_purchase['Product Name']

        # Get content-based similar items
        similar_items = get_content_similar_items(
            last_product_id, self.product_indices, self.cosine_sim, self.products, top_n * 2
        )

        if similar_items.empty or 'Product ID' not in similar_items.columns:
            return pd.DataFrame(columns=['Product ID', 'Product Name', 'Category', 'Sub-Category'])

        # Exclude already purchased
        similar_items = similar_items[~similar_items['Product ID'].isin(product_ids)]

        result = similar_items.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category']]
        logger.info(f"Generated {len(result)} content-based recommendations for customer {customer_id}")
        
        return result


__all__ = ['ContentBasedRecommender']
