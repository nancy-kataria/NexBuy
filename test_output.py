from recommendation_func import recommend_popular, recommend_hybrid, recommend_collaborative, recommend_content_based
from data_preprocess import superstore_data, train_df
from helpers import get_customer_orders_and_products, get_customer_preferences

def print_recommendation_output(customer_id, num_recommendations=5):
    print("=" * 60)
    print("Example: Personalized Recommendations for One Customer")
    print("=" * 60)

    # Step 1: Show context
    order_history, product_ids = get_customer_orders_and_products(customer_id, train_df)

    if order_history.empty:
        print(f"\nNo purchase history found for customer '{customer_id}'. Showing global popular items instead.")
        print(recommend_popular(customer_id=None, top_n=num_recommendations))
        return

    print(f"\n Purchase History Summary for Customer: {customer_id}")
    print(f"  - Total Unique Products Purchased: {len(product_ids)}")

    # Last purchase
    last_purchase = order_history.sort_values('Order Date', ascending=False).iloc[0]
    print(f"  - Most Recent Purchase: '{last_purchase['Product Name']}' on {last_purchase['Order Date'].date()} — {last_purchase['Category']} / {last_purchase['Sub-Category']}")

    # Frequent items
    freq_counts = order_history['Product ID'].value_counts()
    top_freq_ids = freq_counts.head(3).index.tolist()
    print("  - Most Frequently Purchased Items:")
    for pid in top_freq_ids:
        row = superstore_data[superstore_data['Product ID'] == pid].iloc[0]
        print(f"    → '{row['Product Name']}' ({freq_counts[pid]} times) — {row['Category']} / {row['Sub-Category']}")

    # Category preferences
    top_cats, top_subcats = get_customer_preferences(customer_id, superstore_data)
    print(f"  - Top Categories: {', '.join(top_cats[:3])}")
    print(f"  - Top Sub-Categories: {', '.join(top_subcats[:3])}")

    print("\n Customer Purchase History")
    history_sorted = order_history.sort_values('Order Date', ascending=False).copy()
    for _, row in history_sorted.iterrows():
        print(f"  [{row['Order Date'].date()}] {row['Product Name']} (ID: {row['Product ID']}) — {row['Category']} / {row['Sub-Category']}")

    print("\n" + "=" * 60)
    print(" Recommendation Outputs")
    print("=" * 60)

    def explain_recommendations(name, df, context_col=None):
        print(f"\nTop {num_recommendations} {name} Recommendations:")
        if context_col:
            print(f"(Based on {context_col})")
        for _, row in df.iterrows():
            reason = []
            if 'Similarity Score' in row and row['Similarity Score'] == 0:
                reason.append("fallback (popular item)")
            elif 'Similarity Score' in row:
                reason.append(f"similarity score: {row['Similarity Score']:.4f}")
            if row['Category'] in top_cats:
                reason.append(f"matches favorite category: {row['Category']}")
            if row['Sub-Category'] in top_subcats:
                reason.append(f"matches frequent sub-category: {row['Sub-Category']}")
            explanation = "; ".join(reason)
            print(f"→ {row['Product Name']} (ID: {row['Product ID']}) — {row['Category']} / {row['Sub-Category']}")
            if explanation:
                print(f"   Explanation: {explanation}\n")

    # Generate all recommendations
    popular_df = recommend_popular(customer_id, top_n=num_recommendations)
    content_df = recommend_content_based(customer_id, top_n=num_recommendations)
    collab_df = recommend_collaborative(customer_id, top_n=num_recommendations)
    hybrid_df = recommend_hybrid(customer_id, top_n=num_recommendations, show_debug=True)

    # Print all with explanations
    explain_recommendations("Popular", popular_df, context_col="overall purchase frequency across all users")
    explain_recommendations("Content-Based", content_df, context_col="last product purchased")
    explain_recommendations("Collaborative", collab_df, context_col="co-purchase patterns of similar users")
    explain_recommendations("Hybrid", hybrid_df, context_col="content + collaborative + popularity")

# Example usage:
print_recommendation_output("WB-21850", num_recommendations=5)