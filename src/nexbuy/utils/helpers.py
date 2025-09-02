"""
Helper functions for the NexBuy recommendation system
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


def get_customer_orders_and_products(customer_id: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fetches purchase data and unique purchased product IDs for a customer.

    Args:
        customer_id (str): The ID of the target customer.
        df (pd.DataFrame): The main DataFrame containing all transaction data.
                           Must include 'Customer ID' and 'Product ID'.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing:
            - order_history (pd.DataFrame): A DataFrame filtered to only include
                                            rows for the given customer_id. Returns
                                            an empty DataFrame if customer not found.
            - product_ids (np.ndarray): A NumPy array of unique Product IDs
                                          purchased by the customer. Returns an
                                          empty array if customer not found.
    """
    order_history = df[df['Customer ID'] == customer_id].copy()
    # Using .copy() is good practice here to prevent potential SettingWithCopyWarning
    # if the returned DataFrame is modified later in another function.
    product_ids = order_history['Product ID'].unique()
    return order_history, product_ids


def get_unseen_products(customer_id: str, df: pd.DataFrame, product_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Get a list of products the customer hasn't purchased yet

    Args:
        customer_id (str): ID of the target customer.
        df (pd.DataFrame): Full transaction data (e.g., superstore_data)
                           used to find customer history.
        product_df (pd.DataFrame): DataFrame of all products to recommend
                                   from (e.g., product_popularity).

    Returns:
        tuple[pd.DataFrame, np.ndarray]: 
            - filtered product_df with only unseen products
            - list of purchased Product IDs for fallback logic
    """
    _, product_ids = get_customer_orders_and_products(customer_id, df)
    return product_df[~product_df['Product ID'].isin(product_ids)], product_ids


def add_fallback_if_needed(recommendations: pd.DataFrame, product_ids: np.ndarray, 
                          product_df: pd.DataFrame, n: int, by: str) -> pd.DataFrame:
    """
    Add fallback recommendations if there aren't enough unseen products to recommend.
    This uses globally popular products (based on 'Quantity' or 'Sales') to fill the gap.

    Args:
        recommendations (pd.DataFrame): filtered list of unseen, ranked products
        product_ids (np.ndarray): list of already purchased product IDs
        product_df (pd.DataFrame): global product list (e.g., product_popularity)
        n (int): number of products we want to recommend
        by (str): popularity metric ('Quantity' or 'Sales')

    Returns:
        pd.DataFrame: final DataFrame of n recommendations
    """
    # Import here to avoid circular imports
    from .calculations import get_global_popular_products
    
    if len(recommendations) < n:
        # Fallback to global popular items if needed
        fallback = get_global_popular_products(product_df, top_n=n, by=by)
        fallback = fallback[~fallback['Product ID'].isin(product_ids)]
        recommendations = pd.concat([recommendations, fallback]).drop_duplicates('Product ID')
    return recommendations


def get_customer_preferences(customer_id: str, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Gets the customer's most frequent categories and sub-categories.

    Analyzes a customer's purchase history to find the categories and
    sub-categories they interact with most often, based on the count
    of purchases in each. Used for personalized popularity recommendations.

    Args:
        customer_id (str): The ID of the target customer.
        df (pd.DataFrame): The DataFrame containing transaction data, including
                           'Customer ID', 'Category', and 'Sub-Category' columns.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - The first list contains category names, sorted by frequency (most frequent first).
            - The second list contains sub-category names, sorted by frequency.
            Returns two empty lists ([], []) if the customer has no purchase history in df.
    """
    order_history, _ = get_customer_orders_and_products(customer_id, df)
    if order_history.empty:
        return [], []
    top_categories = order_history['Category'].value_counts().index.tolist()
    top_subcategories = order_history['Sub-Category'].value_counts().index.tolist()
    return top_categories, top_subcategories


__all__ = [
    'get_customer_orders_and_products',
    'get_unseen_products', 
    'add_fallback_if_needed',
    'get_customer_preferences'
]
