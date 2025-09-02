"""
NexBuy Streamlit App using the NEW modular structure
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add multiple path options for deployment flexibility
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, 'src'),
    current_dir,
    os.path.join(current_dir, '..'),
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Try different import strategies
import_success = False
error_messages = []

# Strategy 1: Direct import from src
try:
    from nexbuy.data.preprocessing import train_df, test_df
    from nexbuy.models.popular import PopularRecommender
    from nexbuy.models.content_based import ContentBasedRecommender
    from nexbuy.models.collaborative import CollaborativeRecommender
    from nexbuy.models.hybrid import HybridRecommender
    from nexbuy.utils.precomputations import (
        product_popularity, product_indices, cosine_sim, products, product_similarity_df
    )
    from nexbuy.evaluation.metrics import precision_at_k
    import_success = True
except ImportError as e:
    error_messages.append(f"Strategy 1 failed: {e}")

# Strategy 2: Import from src.nexbuy
if not import_success:
    try:
        from src.nexbuy.data.preprocessing import train_df, test_df
        from src.nexbuy.models.popular import PopularRecommender
        from src.nexbuy.models.content_based import ContentBasedRecommender
        from src.nexbuy.models.collaborative import CollaborativeRecommender
        from src.nexbuy.models.hybrid import HybridRecommender
        from src.nexbuy.utils.precomputations import (
            product_popularity, product_indices, cosine_sim, products, product_similarity_df
        )
        from src.nexbuy.evaluation.metrics import precision_at_k
        import_success = True
    except ImportError as e:
        error_messages.append(f"Strategy 2 failed: {e}")

if not import_success:
    st.error("❌ Failed to import NexBuy modules")
    st.error("Deployment structure issues detected:")
    for msg in error_messages:
        st.error(msg)
    
    st.info("""
    **Debugging Information:**
    - Current working directory: {}
    - Python path: {}
    - Files in current directory: {}
    """.format(
        os.getcwd(),
        sys.path[:3],
        [f for f in os.listdir('.') if not f.startswith('.')]
    ))
    st.stop()

st.set_page_config(page_title="NexBuy", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 NexBuy - Multi-Strategy Product Recommendation System")

# Initialize recommenders
@st.cache_resource
def load_recommenders():
    """Load and fit all recommenders"""
    
    # Popular recommender
    popular_rec = PopularRecommender(by='Quantity')
    popular_rec.fit(train_df)
    
    # Content-based recommender
    content_rec = ContentBasedRecommender()
    content_rec.fit(train_df, product_indices, cosine_sim, products, product_popularity)
    
    # Collaborative recommender
    collab_rec = CollaborativeRecommender()
    collab_rec.fit(train_df, product_similarity_df, product_popularity)
    
    # Hybrid recommender
    hybrid_rec = HybridRecommender(w_content=0.4, w_collab=0.4, w_pop=0.2)
    hybrid_rec.fit(train_df, product_indices, cosine_sim, products, product_similarity_df, product_popularity)
    
    return {
        'popular': popular_rec,
        'content': content_rec,
        'collaborative': collab_rec,
        'hybrid': hybrid_rec
    }

recommenders = load_recommenders()

# Sidebar for individual recommendations
st.sidebar.title("🎯 Try Individual Recommendations")
customer_id = st.sidebar.text_input("Customer ID", value="WB-21850")
top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

if st.sidebar.button("Get Recommendations"):
    if customer_id:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Popular Recommendations")
            try:
                popular_recs = recommenders['popular'].recommend(customer_id, top_n)
                st.dataframe(popular_recs, use_container_width=True)
            except Exception as e:
                st.error(f"Error getting popular recommendations: {e}")
            
            st.subheader("🤝 Collaborative Recommendations")
            try:
                collab_recs = recommenders['collaborative'].recommend(customer_id, top_n)
                st.dataframe(collab_recs, use_container_width=True)
            except Exception as e:
                st.error(f"Error getting collaborative recommendations: {e}")
        
        with col2:
            st.subheader("📄 Content-Based Recommendations")
            try:
                content_recs = recommenders['content'].recommend(customer_id, top_n)
                st.dataframe(content_recs, use_container_width=True)
            except Exception as e:
                st.error(f"Error getting content-based recommendations: {e}")
            
            st.subheader("🚀 Hybrid Recommendations")
            try:
                hybrid_recs = recommenders['hybrid'].recommend(customer_id, top_n)
                st.dataframe(hybrid_recs, use_container_width=True)
            except Exception as e:
                st.error(f"Error getting hybrid recommendations: {e}")

# --- Product ID based recommendations ---
st.sidebar.markdown("---")
st.sidebar.title("🧪 Try Recommendations by Product ID")
product_id = st.sidebar.text_input("Product ID", value="FUR-BO-10001798")

if st.sidebar.button("Get Similar Products"):
    pid = product_id.strip()
    if not pid:
        st.warning("Please enter a valid Product ID.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Content-Based: Similar Products")
            # Content similarity via precomputed matrices
            if pid in product_indices:
                try:
                    idx = product_indices[pid]
                    # Get similarity scores and order
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    top = [i for i, _ in sim_scores[1: top_n + 1]]  # exclude the item itself
                    content_similar = products.iloc[top][['Product ID', 'Product Name', 'Category', 'Sub-Category']].copy()
                    content_similar.insert(0, 'Source Product ID', pid)
                    st.dataframe(content_similar, use_container_width=True)
                except Exception as e:
                    st.error(f"Content-based similarity unavailable: {e}")
            else:
                st.info("Product not found in content index. Try another Product ID.")

        with col2:
            st.subheader("🤝 Collaborative: Similar Products")
            if 'product_similarity_df' in globals() and pid in product_similarity_df.columns:
                try:
                    scores = product_similarity_df[pid].drop(labels=[pid], errors='ignore').sort_values(ascending=False)
                    top_ids = scores.head(top_n).index.tolist()
                    collab_similar = product_popularity[product_popularity['Product ID'].isin(top_ids)][['Product ID', 'Product Name', 'Category', 'Sub-Category']].copy()
                    # Preserve order according to similarity
                    collab_similar['__order'] = collab_similar['Product ID'].apply(lambda x: top_ids.index(x) if x in top_ids else len(top_ids))
                    collab_similar = collab_similar.sort_values('__order').drop(columns='__order')
                    collab_similar.insert(0, 'Source Product ID', pid)
                    st.dataframe(collab_similar, use_container_width=True)
                except Exception as e:
                    st.error(f"Collaborative similarity unavailable: {e}")
            else:
                st.info("Product not found in collaborative similarity data. Try another Product ID.")

# Main evaluation interface
st.subheader("📊 Model Evaluation")
k = st.slider("Select value of K for evaluation", 1, 10, 5)

if st.button("🔥 Evaluate All Models"):
    with st.spinner("Running comprehensive evaluation on test set..."):
        try:
            # Create wrapper functions for evaluation
            def recommend_popular_wrapper(customer_id, top_n=5):
                return recommenders['popular'].recommend(customer_id, top_n)
            
            def recommend_content_wrapper(customer_id, top_n=5):
                return recommenders['content'].recommend(customer_id, top_n)
            
            def recommend_collab_wrapper(customer_id, top_n=5):
                return recommenders['collaborative'].recommend(customer_id, top_n)
            
            def recommend_hybrid_wrapper(customer_id, top_n=5):
                return recommenders['hybrid'].recommend(customer_id, top_n)
            
            # Evaluate all methods
            prec_pop = precision_at_k(recommend_popular_wrapper, test_df, k=k, method_name="Popular (New)")
            prec_cont = precision_at_k(recommend_content_wrapper, test_df, k=k, method_name="Content-Based (New)")
            prec_collab = precision_at_k(recommend_collab_wrapper, test_df, k=k, method_name="Collaborative (New)")
            prec_hybrid = precision_at_k(recommend_hybrid_wrapper, test_df, k=k, method_name="Hybrid (New)")
            
            scores = {
                '🏆 Popular': prec_pop,
                '📄 Content-Based': prec_cont,
                '🤝 Collaborative': prec_collab,
                '🚀 Hybrid': prec_hybrid
            }
            
            df = pd.DataFrame.from_dict(scores, orient='index', columns=[f'Precision@{k}'])
            
            st.success("✅ Evaluation completed!")
            
            # Show results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(f"📊 Precision@{k} Results")
                st.table(df.style.format({f'Precision@{k}': '{:.4f}'}))
                
                # Highlight best performing model
                best_model = df.idxmax()[0]
                best_score = df.max()[0]
                st.info(f"🥇 **Best Model**: {best_model} with Precision@{k} = {best_score:.4f}")
            
            with col2:
                st.subheader("📈 Performance Comparison")
                st.bar_chart(df)
                
                # Show model comparison
                st.subheader("🔍 Algorithm Comparison")
                comparison_df = pd.DataFrame({
                    'Algorithm': ['Popular', 'Content-Based', 'Collaborative', 'Hybrid'],
                    'Type': ['Popularity', 'Content Similarity', 'User Behavior', 'Combined'],
                    'Precision@5': [prec_pop, prec_cont, prec_collab, prec_hybrid]
                })
                st.dataframe(comparison_df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error during evaluation: {e}")
