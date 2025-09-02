"""
NexBuy Streamlit App using the NEW modular structure
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import from NEW structure
from nexbuy.data.preprocessing import train_df, test_df
from nexbuy.models.popular import PopularRecommender
from nexbuy.models.content_based import ContentBasedRecommender
from nexbuy.models.collaborative import CollaborativeRecommender
from nexbuy.models.hybrid import HybridRecommender
from nexbuy.utils.precomputations import (
    product_popularity, product_indices, cosine_sim, products, product_similarity_df
)
from nexbuy.evaluation.metrics import precision_at_k

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
            popular_recs = recommenders['popular'].recommend(customer_id, top_n)
            st.dataframe(popular_recs, use_container_width=True)
            
            st.subheader("🤝 Collaborative Recommendations")
            collab_recs = recommenders['collaborative'].recommend(customer_id, top_n)
            st.dataframe(collab_recs, use_container_width=True)
        
        with col2:
            st.subheader("📄 Content-Based Recommendations")
            content_recs = recommenders['content'].recommend(customer_id, top_n)
            st.dataframe(content_recs, use_container_width=True)
            
            st.subheader("🚀 Hybrid Recommendations")
            hybrid_recs = recommenders['hybrid'].recommend(customer_id, top_n)
            st.dataframe(hybrid_recs, use_container_width=True)

# Main evaluation interface
st.subheader("📊 Model Evaluation")
k = st.slider("Select value of K for evaluation", 1, 10, 5)

if st.button("🔥 Evaluate All Models"):
    with st.spinner("Running comprehensive evaluation on test set..."):
        
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
