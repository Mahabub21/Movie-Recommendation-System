from pathlib import Path

import streamlit as st

from src.content_recommender import ContentRecommender
from src.data_pipeline import clean_and_prepare_data


st.set_page_config(page_title="Movie Recommender", layout="wide")


@st.cache_data
def load_data(base_dir: Path):
    data = clean_and_prepare_data(base_dir / "dataset")
    return data["movies"], data["ratings"]


@st.cache_resource
def build_model(movies_df):
    return ContentRecommender(min_df=1).fit(movies_df)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    movies, ratings = load_data(base_dir)
    model = build_model(movies)

    avg_ratings = (
        ratings.groupby("movieId")["rating"]
        .mean()
        .reset_index(name="avg_rating")
    )

    st.title("Movie Recommender")
    st.caption("No .pkl file required. Model is built from dataset automatically.")

    movie_options = sorted(movies["title"].dropna().unique().tolist())
    selected_movie = st.selectbox("Select a movie", movie_options)
    num_recommendations = st.slider("Number of recommendations", min_value=5, max_value=10, value=10)

    if st.button("Recommend"):
        recs = model.recommend(selected_movie, top_n=num_recommendations)
        if recs.empty:
            st.warning("Movie not found.")
        else:
            st.subheader("Recommended Movies")
            recs = recs.merge(avg_ratings, on="movieId", how="left")
            recs["avg_rating"] = recs["avg_rating"].round(2)
            st.dataframe(
                recs[["movieId", "title", "genres", "avg_rating", "similarity"]],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()