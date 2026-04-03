from pathlib import Path

import joblib

from src.collaborative_recommender import predict_rating, recommend_for_user, train_svd_model
from src.content_recommender import ContentRecommender
from src.data_pipeline import clean_and_prepare_data, get_popular_movies, get_top_rated_movies
from src.visualize import plot_rating_distribution, plot_top_genres, plot_user_movie_heatmap


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "dataset"
    artifacts_dir = base_dir / "artifacts"
    plots_dir = base_dir / "plots"

    artifacts_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("Loading and cleaning data...")
    data = clean_and_prepare_data(data_dir)
    movies = data["movies"]
    ratings = data["ratings"]

    print("\nTop 10 Most Rated Movies:")
    top_popular = get_popular_movies(movies, ratings, top_n=10)
    print(top_popular[["title", "rating_count"]])
    top_popular.to_csv(artifacts_dir / "top_popular_movies.csv", index=False)

    print("\nTop 10 Highest Rated Movies (min 50 ratings):")
    top_rated = get_top_rated_movies(movies, ratings, min_ratings=50, top_n=10)
    print(top_rated[["title", "mean", "count"]])
    top_rated.to_csv(artifacts_dir / "top_rated_movies.csv", index=False)

    print("\nTraining content-based recommender...")
    content_model = ContentRecommender(min_df=1).fit(movies)
    joblib.dump(content_model, artifacts_dir / "content_model.pkl")

    for query in ["Inception", "The Matrix"]:
        print(f"\nRecommendations for '{query}':")
        recs = content_model.recommend(query, top_n=10)
        if recs.empty:
            print("No movie found with that title.")
        else:
            print(recs[["title", "similarity"]])

    print("\nGenerating visualizations...")
    plot_rating_distribution(ratings, plots_dir / "rating_distribution.png")
    plot_top_genres(movies, plots_dir / "top_genres.png")
    plot_user_movie_heatmap(ratings, plots_dir / "user_movie_heatmap.png")
    print("Saved plots in 'plots/' directory.")

    print("\nTraining collaborative filtering model (SVD)...")
    svd_model, rmse = train_svd_model(ratings)
    print(f"Collaborative Filtering RMSE ({svd_model['backend']} backend): {rmse:.4f}")
    joblib.dump(svd_model, artifacts_dir / "svd_model.pkl")

    example_user = int(ratings["userId"].iloc[0])
    example_movie = int(movies["movieId"].iloc[0])
    prediction = predict_rating(svd_model, user_id=example_user, movie_id=example_movie)
    print(
        f"Predicted rating for user {example_user} on movie {example_movie}: {prediction:.2f}"
    )

    user_recs = recommend_for_user(
        svd_model,
        ratings_df=ratings,
        movies_df=movies,
        user_id=example_user,
        top_n=10,
    )
    user_recs.to_csv(artifacts_dir / "collaborative_recommendations.csv", index=False)

    print("\nProject run complete.")
    print("Artifacts saved in 'artifacts/' and plots in 'plots/'.")


if __name__ == "__main__":
    main()
