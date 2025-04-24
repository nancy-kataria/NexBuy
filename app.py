import streamlit as st
import pandas as pd


from recommendation_func import recommend_content_based, recommend_collaborative, recommend_hybrid, recommend_popular
from data_preprocess import test_df
from evaluation import precision_at_k

st.set_page_config(page_title="Live Evaluation", layout="centered")
st.title("📊 Live Evaluation of Recommendation Models")
st.markdown("This evaluates Precision@k using train-test split.")

k = st.slider("Select value of K", 1, 10, 5)

if st.button("Evaluate All Models"):
    with st.spinner("Running evaluation on test set..."):

        # Evaluate each method
        prec_pop = precision_at_k(recommend_popular, test_df, k=k, method_name="Popular")
        prec_cont = precision_at_k(recommend_content_based, test_df, k=k, method_name="Content-Based")
        prec_collab = precision_at_k(recommend_collaborative, test_df, k=k, method_name="Collaborative")
        prec_hybrid = precision_at_k(recommend_hybrid, test_df, k=k, method_name="Hybrid")

        scores = {
            'Popular': prec_pop,
            'Content-Based': prec_cont,
            'Collaborative': prec_collab,
            'Hybrid': prec_hybrid
        }

        df = pd.DataFrame.from_dict(scores, orient='index', columns=[f'Precision@{k}'])

        st.success("✅ Evaluation completed!")
        st.subheader(f"Precision@{k} Scores")
        st.table(df)

        st.subheader("📊 Comparison Chart")
        st.bar_chart(df)

else:
    st.info("Click the button above to evaluate all recommendation models on test users.")
