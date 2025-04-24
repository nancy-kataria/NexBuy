from recommendation_func import recommend_collaborative, recommend_content_based, recommend_hybrid, recommend_popular
from data_preprocess import test_df

def precision_at_k(recommend_func, test_df, k=5, method_name=""):
    hits = 0
    total = 0
    user_ids = test_df['Customer ID'].unique()

    for customer_id in user_ids:
        # Actual products the customer bought in the test set
        true_products = set(test_df[test_df['Customer ID'] == customer_id]['Product ID'])

        # Get recommendations using the selected strategy
        try:
            recs = recommend_func(customer_id, top_n=k)
        except TypeError:
            # For functions like recommend_popular that use 'n' instead of 'top_n'
            recs = recommend_func(customer_id=customer_id, n=k)

        if recs is None or recs.empty:
            continue

        recommended_ids = set(recs['Product ID'])

        hits += len(recommended_ids & true_products)
        total += k

    precision = hits / total if total > 0 else 0
    print(f"Precision@{k} for {method_name}: {precision:.4f}")
    return precision

precision_at_k(recommend_popular, test_df, k=5, method_name="Popular")
precision_at_k(recommend_content_based, test_df, k=5, method_name="Content-Based")
precision_at_k(recommend_collaborative, test_df, k=5, method_name="Collaborative")
precision_at_k(recommend_hybrid, test_df, k=5, method_name="Hybrid")
