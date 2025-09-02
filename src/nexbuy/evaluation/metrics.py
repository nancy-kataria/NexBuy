"""
Evaluation metrics for the NexBuy recommendation system
"""
import pandas as pd
from typing import Callable, Set, Optional
import logging

logger = logging.getLogger(__name__)


def precision_at_k(recommend_func: Callable, test_df: pd.DataFrame, 
                  k: int = 5, method_name: str = "") -> float:
    """
    Calculate Precision@K for a recommendation function.
    
    Args:
        recommend_func: Function that takes customer_id and top_n as parameters
        test_df: DataFrame containing test data with actual purchases
        k: Number of top recommendations to consider
        method_name: Name of the method for logging
        
    Returns:
        float: Precision@K score
    """
    hits = 0
    total = 0
    user_ids = test_df['Customer ID'].unique()
    
    logger.info(f"Evaluating {method_name} with Precision@{k} on {len(user_ids)} users")
    
    for customer_id in user_ids:
        # Actual products the customer bought in the test set
        true_products = set(test_df[test_df['Customer ID'] == customer_id]['Product ID'])
        
        # Get recommendations using the selected strategy
        try:
            recs = recommend_func(customer_id, top_n=k)
        except TypeError:
            # For functions like recommend_popular that use 'n' instead of 'top_n'
            try:
                recs = recommend_func(customer_id=customer_id, n=k)
            except TypeError:
                # Try with just positional arguments
                recs = recommend_func(customer_id, k)
        except Exception as e:
            logger.warning(f"Error generating recommendations for customer {customer_id}: {e}")
            continue
        
        if recs is None or recs.empty:
            continue
            
        if 'Product ID' not in recs.columns:
            logger.warning(f"No 'Product ID' column in recommendations for {customer_id}")
            continue
        
        recommended_ids = set(recs['Product ID'])
        
        hits += len(recommended_ids & true_products)
        total += k
    
    precision = hits / total if total > 0 else 0
    logger.info(f"Precision@{k} for {method_name}: {precision:.4f}")
    return precision


def recall_at_k(recommend_func: Callable, test_df: pd.DataFrame, 
               k: int = 5, method_name: str = "") -> float:
    """
    Calculate Recall@K for a recommendation function.
    
    Args:
        recommend_func: Function that takes customer_id and top_n as parameters
        test_df: DataFrame containing test data with actual purchases
        k: Number of top recommendations to consider
        method_name: Name of the method for logging
        
    Returns:
        float: Recall@K score
    """
    total_relevant_items = 0
    relevant_recommended = 0
    user_ids = test_df['Customer ID'].unique()
    
    logger.info(f"Evaluating {method_name} with Recall@{k} on {len(user_ids)} users")
    
    for customer_id in user_ids:
        # Actual products the customer bought in the test set
        true_products = set(test_df[test_df['Customer ID'] == customer_id]['Product ID'])
        total_relevant_items += len(true_products)
        
        # Get recommendations
        try:
            recs = recommend_func(customer_id, top_n=k)
        except Exception as e:
            logger.warning(f"Error generating recommendations for customer {customer_id}: {e}")
            continue
        
        if recs is None or recs.empty or 'Product ID' not in recs.columns:
            continue
        
        recommended_ids = set(recs['Product ID'])
        relevant_recommended += len(recommended_ids & true_products)
    
    recall = relevant_recommended / total_relevant_items if total_relevant_items > 0 else 0
    logger.info(f"Recall@{k} for {method_name}: {recall:.4f}")
    return recall


def evaluate_recommender(recommend_func: Callable, test_df: pd.DataFrame,
                        k_values: list = [1, 3, 5, 10], method_name: str = "") -> pd.DataFrame:
    """
    Comprehensive evaluation of a recommender function.
    
    Args:
        recommend_func: Function that takes customer_id and top_n as parameters
        test_df: DataFrame containing test data
        k_values: List of k values to evaluate
        method_name: Name of the method for reporting
        
    Returns:
        pd.DataFrame: DataFrame with evaluation results
    """
    results = []
    
    for k in k_values:
        precision = precision_at_k(recommend_func, test_df, k, method_name)
        recall = recall_at_k(recommend_func, test_df, k, method_name)
        
        results.append({
            'Method': method_name,
            'K': k,
            'Precision@K': precision,
            'Recall@K': recall,
            'F1@K': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        })
    
    return pd.DataFrame(results)


__all__ = ['precision_at_k', 'recall_at_k', 'evaluate_recommender']
