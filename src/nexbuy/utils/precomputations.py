"""
Precomputations module for the NexBuy recommendation system.
Handles all the heavy computations that can be done once and reused.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class PrecomputationEngine:
    """Handles all precomputations for the recommendation system."""
    
    def __init__(self, superstore_data: pd.DataFrame):
        self.superstore_data = superstore_data
        self.product_popularity = None
        self.products = None
        self.cosine_sim = None
        self.product_indices = None
        self.product_similarity_df = None
        self.user_product_matrix = None
        
    def compute_product_popularity(self) -> pd.DataFrame:
        """Compute product popularity metrics."""
        logger.info("Computing product popularity...")
        
        self.product_popularity = self.superstore_data.groupby('Product ID').agg({
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
        
        logger.info(f"Computed popularity for {len(self.product_popularity)} products")
        return self.product_popularity
    
    def compute_content_similarity(self) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        """Compute content-based similarity using TF-IDF."""
        logger.info("Computing content-based similarity...")
        
        # Prepare content information
        self.superstore_data['product_info'] = (
            self.superstore_data['Product Name'].astype(str) + ' ' +
            self.superstore_data['Category'].astype(str) + ' ' +
            self.superstore_data['Sub-Category'].astype(str)
        )
        
        # One row per product
        self.products = self.superstore_data.drop_duplicates(subset='Product ID')[
            ['Product ID', 'Product Name', 'Category', 'Sub-Category', 'product_info']
        ].reset_index(drop=True)
        
        # TF-IDF Matrix and Cosine Similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.products['product_info'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Product Index Mapping
        self.product_indices = pd.Series(
            self.products.index, 
            index=self.products['Product ID']
        ).drop_duplicates()
        
        logger.info(f"Computed content similarity for {len(self.products)} products")
        return self.cosine_sim, self.product_indices, self.products
    
    def compute_collaborative_similarity(self) -> pd.DataFrame:
        """Compute collaborative filtering similarity."""
        logger.info("Computing collaborative similarity...")
        
        # Create user-product interaction matrix
        self.user_product_matrix = self.superstore_data.pivot_table(
            index='Customer ID',
            columns='Product ID',
            values='Quantity',
            aggfunc='sum'
        ).fillna(0)
        
        # Compute item-item similarity
        product_similarity = cosine_similarity(self.user_product_matrix.T)
        
        # Store as DataFrame
        self.product_similarity_df = pd.DataFrame(
            product_similarity,
            index=self.user_product_matrix.columns,
            columns=self.user_product_matrix.columns
        )
        
        logger.info(f"Computed collaborative similarity for {len(self.user_product_matrix.columns)} products")
        return self.product_similarity_df
    
    def compute_all(self) -> dict:
        """Compute all precomputations and return as dictionary."""
        logger.info("Starting all precomputations...")
        
        self.compute_product_popularity()
        self.compute_content_similarity()
        self.compute_collaborative_similarity()
        
        return {
            'product_popularity': self.product_popularity,
            'products': self.products,
            'cosine_sim': self.cosine_sim,
            'product_indices': self.product_indices,
            'product_similarity_df': self.product_similarity_df,
            'user_product_matrix': self.user_product_matrix
        }
    
    def get_computations(self) -> dict:
        """Get all computed data."""
        return {
            'product_popularity': self.product_popularity,
            'products': self.products,
            'cosine_sim': self.cosine_sim,
            'product_indices': self.product_indices,
            'product_similarity_df': self.product_similarity_df,
            'user_product_matrix': self.user_product_matrix
        }


# For backward compatibility, we'll import the data and compute everything
from ..data.preprocessing import superstore_data

# Initialize and compute
engine = PrecomputationEngine(superstore_data)
computations = engine.compute_all()

# Export individual components for backward compatibility
product_popularity = computations['product_popularity']
products = computations['products']
cosine_sim = computations['cosine_sim']
product_indices = computations['product_indices']
product_similarity_df = computations['product_similarity_df']
user_product_matrix = computations['user_product_matrix']

__all__ = [
    'PrecomputationEngine',
    'engine',
    'product_popularity',
    'products', 
    'cosine_sim',
    'product_indices',
    'product_similarity_df',
    'user_product_matrix'
]
