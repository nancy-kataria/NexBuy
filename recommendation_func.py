from helpers import get_customer_orders_and_products, get_global_popular_products, get_unseen_products
from data_preprocess import train_df
from calculations import get_content_similar_items
from precomputations import product_popularity, product_similarity_df, product_indices, cosine_sim, products
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helpers import add_fallback_if_needed
import numpy as np

# === Main Recommendation Functions ===
def recommend_popular(customer_id=None, top_n=5, by='Quantity'):
    """
    Recommends popular products, optionally personalized for a customer.

    Modes:
    1. Global: If customer_id is None, returns globally popular products.
    2. Unseen for Customer: Returns globally popular products not yet purchased by the customer,
       with fallback if fewer than n are found.

    Args:
        customer_id (str, optional): The ID of the customer. Defaults to None.
        top_n (int, optional): The number of products to recommend. Defaults to 10.
        by (str, optional): The metric for popularity ('Quantity' or 'Sales').
                            Defaults to 'Quantity'.

    Returns:
        pd.DataFrame: Top-N recommended products.

    """
    if by not in ['Quantity', 'Sales']:
        raise ValueError("Parameter 'by' must be either 'Quantity' or 'Sales'")

    # Case 1: No customer → return top global products
    if customer_id is None:
        print("No customer ID provided. Returning global popular products.")
        return get_global_popular_products(top_n=top_n)

    # Case 2: Exclude products already purchased
    unseen_products, product_ids = get_unseen_products(customer_id, train_df , product_popularity)
    unseen_products = unseen_products.sort_values(by=by, ascending=False)

    # Apply fallback if needed
    final = add_fallback_if_needed(unseen_products, product_ids, product_popularity, top_n, by)

    return final.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category', by]]

def recommend_content_based(customer_id, top_n=5):
    """
    Recommends products similar to the last item purchased by a customer.

    Finds the customer's most recent purchase and then uses content-based
    similarity (get_content_similar_items) to find similar items.

    Args:
        customer_id (str): The ID of the customer.
        top_n (int, optional): The number of similar products to recommend.
                               Defaults to 5.

    Returns:
        pd.DataFrame or str: A DataFrame containing the recommended products
                             (from get_content_similar_items) or a string message
                             if the customer has no purchase history.
                             (Consider changing the string return to an empty DataFrame
                             for consistency).
    """
    # Case 1: No customer → return top global products
    if customer_id is None:
        print("No customer ID provided. Returning global popular products.")
        return get_global_popular_products(top_n=top_n)

    order_history, product_ids = get_customer_orders_and_products(customer_id, train_df)
    if order_history.empty:
          print(f"No purchase history for customer '{customer_id}'.")
          return pd.DataFrame() # Or an empty list

    # Get last product bought
    last_purchase = order_history.sort_values('Order Date', ascending=False).iloc[0]
    last_product_id = last_purchase['Product ID']
    last_product_name = last_purchase['Product Name']
    print(f"Based on last product purchased (ID: {last_product_id}): {last_product_name}")

    # Get content-based similar items
    similar_items = get_content_similar_items(last_product_id, top_n * 2)  # get more to allow filtering

    if similar_items.empty or 'Product ID' not in similar_items.columns:
        return pd.DataFrame(columns=['Product ID', 'Product Name', 'Category', 'Sub-Category'])

    # Exclude already purchased
    similar_items = similar_items[~similar_items['Product ID'].isin(product_ids)]

    return similar_items.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category']]

def recommend_collaborative(customer_id, top_n=5):
    """Recommends products to a customer based on collaborative filtering.

    Aggregates similarity scores from items the customer has purchased to find
    new items that are similar based on co-purchase patterns across all users.
    Excludes items already purchased by the customer.

    Args:
        customer_id (str): The ID of the customer.
        top_n (int, optional): The number of products to recommend. Defaults to 5.

    Returns:
        pd.DataFrame or str: A DataFrame containing the top_n recommended products
                             with columns ['Product ID', 'Product Name', 'Category',
                             'Sub-Category']. Returns a string message if the customer
                             has no history or suitable product data isn't found.
                             (Consider changing string returns to an empty DataFrame).
    """
    # Case 1: No customer → return top global products
    if customer_id is None:
        print("No customer ID provided. Returning global popular products.")
        return get_global_popular_products(top_n=top_n)

    order_history, product_ids = get_customer_orders_and_products(customer_id, train_df)
    if order_history.empty:
        print(f"No purchase history for customer '{customer_id}'.")
        return pd.DataFrame() # Or an empty list

    # If user has multiple purchases, accumulate similarity
    total_collab_scores = None
    valid_count = 0
    for pid in product_ids:
        if pid not in product_similarity_df.columns:
            continue
        product_scores = product_similarity_df[pid]
        total_collab_scores = product_scores if total_collab_scores is None else total_collab_scores + product_scores
        valid_count += 1

    if total_collab_scores is None or valid_count == 0:
        print(f"No valid products found for similarity for customer '{customer_id}'.")
        return pd.DataFrame()

    # Normalize if multiple products
    total_collab_scores = total_collab_scores / valid_count

    # Remove already purchased products
    total_collab_scores = total_collab_scores.drop(labels=product_ids, errors='ignore')

    # Get top similar product IDs
    top_scores = total_collab_scores.sort_values(ascending=False)
    top_ids = top_scores.head(top_n).index.tolist()

    # Fetch recommended products
    recommendations = product_popularity[product_popularity['Product ID'].isin(top_ids)].copy()
    recommendations['Similarity Score'] = top_scores[top_ids].values

    # Fallback if not enough items
    if len(recommendations) < top_n:
        print(f"Only {len(recommendations)} collaborative recommendations found. Adding fallback items.")
        fallback = get_global_popular_products(top_n=top_n * 2)  # more to ensure enough
        fallback = fallback[~fallback['Product ID'].isin(product_ids + top_ids)]
        fallback = fallback.head(top_n - len(recommendations))
        fallback['Similarity Score'] = 0  # or None, since fallback isn't similarity-based
        recommendations = pd.concat([recommendations, fallback])

    return recommendations.head(top_n)[['Product ID', 'Product Name', 'Category', 'Sub-Category', 'Similarity Score']]

def recommend_hybrid(customer_id, top_n=5, w_content=0.4, w_collab=0.4, w_pop=0.2, show_debug=False):
    """
    Recommends products using a hybrid approach combining content similarity,
    collaborative similarity, and global popularity.
    """
    # Case 1: No customer → return top global products
    if customer_id is None:
        print("No customer ID provided. Returning global popular products.")
        return get_global_popular_products(top_n=top_n)

    order_history, product_ids = get_customer_orders_and_products(customer_id, train_df)
    if order_history.empty:
        print(f"No purchase history found for customer '{customer_id}'.")
        return pd.DataFrame()

    # --- 1. Calculate Average Content Similarity Scores ---
    purchased_idxs_content = [product_indices[pid] for pid in product_ids if pid in product_indices]
    if not purchased_idxs_content:
        print(f"No purchased products for customer '{customer_id}' found in content product index.")
        # Could potentially proceed without content score or return empty
        avg_content_sim_scores = np.zeros(len(products)) # Assign zero score if no history match
    else:
        # Average similarity to user's purchase history
        valid_idxs = [idx for idx in purchased_idxs_content if idx < cosine_sim.shape[0]]
        if not valid_idxs:
            print(f"No valid content-based product indices for customer '{customer_id}'.")
            avg_content_sim_scores = np.zeros(len(products))
        else:
            avg_content_sim_scores = sum(cosine_sim[idx] for idx in valid_idxs) / len(valid_idxs)

    content_df = pd.DataFrame({
        'Product ID': products['Product ID'], # Use Product ID from the 'products' DataFrame
        'content_score': avg_content_sim_scores
    })

    # --- 2. Calculate Average Collaborative Similarity Scores ---
    total_collab_sim = None
    valid_purchased_ids_count = 0
    for pid in product_ids:
        if pid not in product_similarity_df.columns:
            continue
        product_scores = product_similarity_df[pid]
        total_collab_sim = product_scores if total_collab_sim is None else total_collab_sim + product_scores
        valid_purchased_ids_count += 1

    if total_collab_sim is None:
        print(f"No valid products found for collaborative similarity for customer '{customer_id}'.")
         # Assign zero score if no history match in collaborative matrix
        collab_df = pd.DataFrame({'Product ID': product_similarity_df.columns, 'collab_score': 0.0})
    else:
        avg_collab_sim_scores = total_collab_sim / valid_purchased_ids_count
        collab_df = avg_collab_sim_scores.reset_index()
        collab_df.columns = ['Product ID', 'collab_score']

    # --- 3. Combine All Scores ---
    # Start with all products and their popularity
    combined_df = product_popularity[['Product ID', 'Product Name', 'Category', 'Sub-Category', 'popularity_score']].copy()

    # Merge content scores
    combined_df = combined_df.merge(content_df, on='Product ID', how='left')
    combined_df['content_score'] = combined_df['content_score'].fillna(0)

    # Merge collaborative scores
    combined_df = combined_df.merge(collab_df, on='Product ID', how='left')
    combined_df['collab_score'] = combined_df['collab_score'].fillna(0)

    # Filter out already purchased items
    combined_df = combined_df[~combined_df['Product ID'].isin(product_ids)].copy()

    # --- 4. Normalize All Scores ---
    # if not normalize, popularity score is just too high
    scaler = MinMaxScaler()
    combined_df[['content_score', 'collab_score', 'popularity_score']] = scaler.fit_transform(
        combined_df[['content_score', 'collab_score', 'popularity_score']]
    )

    # --- 5. Calculate Final Score ---
    combined_df['final_score'] = (
        w_content * combined_df['content_score'] +
        w_collab * combined_df['collab_score'] +
        w_pop * combined_df['popularity_score']
    )

    # --- 6. Show Debug Info (Optional) ---
    if show_debug:
        print("\n[DEBUG] Top products by each score (before final sort):")
        print(combined_df[['Product Name', 'content_score', 'collab_score', 'popularity_score', 'final_score']]
              .sort_values(by='final_score', ascending=False).head(10))

    # --- 7. Sort and Return ---
    final_recommendations = combined_df.sort_values(by='final_score', ascending=False).head(top_n)

    # --- 8. Fallback Logic ---
    if len(final_recommendations) < top_n:
      print(f"Only {len(final_recommendations)} hybrid recommendations found. Adding fallback items.")
      fallback = get_global_popular_products(n=top_n * 2)
      fallback = fallback[~fallback['Product ID'].isin(product_ids + final_recommendations['Product ID'].tolist())]
      fallback['final_score'] = 0  # Neutral fallback score
      final_recommendations = pd.concat([final_recommendations, fallback.head(top_n - len(final_recommendations))])

    return final_recommendations[['Product ID', 'Product Name', 'Category', 'Sub-Category', 'final_score']]