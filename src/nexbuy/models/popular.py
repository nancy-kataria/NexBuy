"""
Popular-based recommender for the NexBuy recommendation system
"""
import pandas as pd
from typing import Optional
import logging

from .base import BaseRecommender
from ..utils.helpers import get_unseen_products, add_fallback_if_needed
from ..utils.calculations import get_global_popular_products

logger = logging.getLogger(__name__)


class PopularRecommender(BaseRecommender):
    """
    Popular-based recommendation system.
    
    Recommends globally popular products based on sales or quantity metrics.
    Can be personalized to exclude products already purchased by the customer.
    """
    
    def __init__(self, by: str = 'Quantity'):
        """
        Initialize the popular recommender.
        
        Args:
            by (str): Metric for popularity ('Quantity' or 'Sales')
        """
        super().__init__("PopularRecommender")
        
        if by not in ['Quantity', 'Sales']:
            raise ValueError("Parameter 'by' must be either 'Quantity' or 'Sales'")
        
        self.by = by
        self.product_popularity = None
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame) -> 'PopularRecommender':
        """
        Fit the recommender on training data.
        
        Args:
            train_data (pd.DataFrame): Training transaction data
            
        Returns:
            self: The fitted recommender
        """
        logger.info("Fitting PopularRecommender...")
        
        self.train_data = train_data
        
        # Compute product popularity
        self.product_popularity = train_data.groupby('Product ID').agg({
            'Product Name': 'first',
            'Category': 'first',
            'Sub-Category': 'first',
            'Quantity': 'sum',
            'Sales': 'sum'
        }).reset_index()
        
        # Normalize popularity score
        self.product_popularity['popularity_score'] = (
            self.product_popularity['Quantity'] / self.product_popularity['Quantity'].max()
        )
        
        self.is_fitted = True
        logger.info(f"Fitted PopularRecommender on {len(self.product_popularity)} products")
        return self
    
    def recommend(self, customer_id: Optional[str] = None, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate popular-based recommendations.
        
        Args:
            customer_id (str, optional): Customer ID to personalize for
            top_n (int): Number of recommendations to generate
            **kwargs: Additional parameters (currently unused)
            
        Returns:
            pd.DataFrame: Recommended products
        """
        self._check_fitted()
        self._validate_input(customer_id, top_n)
        
        # Case 1: No customer → return top global products
        if customer_id is None:
            logger.info("No customer ID provided. Returning global popular products.")
            return get_global_popular_products(
                self.product_popularity, 
                top_n=top_n, 
                by=self.by
            )
        
        # Case 2: Exclude products already purchased
        unseen_products, product_ids = get_unseen_products(
            customer_id, 
            self.train_data, 
            self.product_popularity
        )
        unseen_products = unseen_products.sort_values(by=self.by, ascending=False)
        
        # Apply fallback if needed
        final = add_fallback_if_needed(
            unseen_products, 
            product_ids, 
            self.product_popularity, 
            top_n, 
            self.by
        )
        
        result = final.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category', self.by]]
        logger.info(f"Generated {len(result)} popular recommendations for customer {customer_id}")
        
        return result
    
    def get_global_popular(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get globally popular products without customer personalization.
        
        Args:
            top_n (int): Number of products to return
            
        Returns:
            pd.DataFrame: Top popular products
        """
        self._check_fitted()
        return get_global_popular_products(
            self.product_popularity, 
            top_n=top_n, 
            by=self.by
        )


__all__ = ['PopularRecommender']
