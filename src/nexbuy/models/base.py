"""
Base recommender class for the NexBuy recommendation system
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union


class BaseRecommender(ABC):
    """Base class for all recommender systems."""
    
    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> 'BaseRecommender':
        """
        Fit the recommender model on training data.
        
        Args:
            train_data (pd.DataFrame): Training data
            
        Returns:
            self: The fitted recommender
        """
        pass
    
    @abstractmethod
    def recommend(self, customer_id: Optional[str] = None, top_n: int = 5, **kwargs) -> pd.DataFrame:
        """
        Generate recommendations.
        
        Args:
            customer_id (str, optional): Customer ID to generate recommendations for
            top_n (int): Number of recommendations to generate
            **kwargs: Additional parameters specific to the recommender
            
        Returns:
            pd.DataFrame: Recommended products
        """
        pass
    
    def _check_fitted(self):
        """Check if the model is fitted."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before generating recommendations")
    
    def _validate_input(self, customer_id: Optional[str], top_n: int):
        """Validate input parameters."""
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        
        # Additional validation can be added here
    
    def __str__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()


__all__ = ['BaseRecommender']
