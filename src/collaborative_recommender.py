import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_svd_model(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_factors: int = 100,
):
    """Train SVD model and return model with RMSE.

    Uses Surprise SVD when installed. Falls back to scikit-learn TruncatedSVD otherwise.
    """
    try:
        from surprise import Dataset, Reader, SVD, accuracy
        from surprise.model_selection import train_test_split as surprise_split

        min_rating = float(ratings_df["rating"].min())
        max_rating = float(ratings_df["rating"].max())
        reader = Reader(rating_scale=(min_rating, max_rating))

        surprise_data = Dataset.load_from_df(
            ratings_df[["userId", "movieId", "rating"]],
            reader,
        )

        trainset, testset = surprise_split(
            surprise_data,
            test_size=test_size,
            random_state=random_state,
        )

        svd = SVD(n_factors=n_factors, random_state=random_state)
        svd.fit(trainset)
        predictions = svd.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)

        return {"backend": "surprise", "model": svd}, rmse
    except ImportError:
        train_ratings, test_ratings = train_test_split(
            ratings_df,
            test_size=test_size,
            random_state=random_state,
        )

        train_matrix = train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        max_components = max(1, min(train_matrix.shape[0], train_matrix.shape[1]) - 1)
        n_components = min(n_factors, max_components)

        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = svd.fit_transform(train_matrix)
        approx = np.dot(reduced, svd.components_)
        approx_df = pd.DataFrame(approx, index=train_matrix.index, columns=train_matrix.columns)

        global_mean = float(ratings_df["rating"].mean())
        preds = []
        truths = []
        for row in test_ratings.itertuples():
            truths.append(float(row.rating))
            if row.userId in approx_df.index and row.movieId in approx_df.columns:
                preds.append(float(approx_df.loc[row.userId, row.movieId]))
            else:
                preds.append(global_mean)

        rmse = float(np.sqrt(mean_squared_error(truths, preds)))

        return {
            "backend": "sklearn",
            "approx_df": approx_df,
            "global_mean": global_mean,
        }, rmse


def predict_rating(model, user_id: int, movie_id: int) -> float:
    """Predict user rating for a movie using trained SVD model."""
    backend = model.get("backend")
    if backend == "surprise":
        return float(model["model"].predict(uid=user_id, iid=movie_id).est)

    approx_df = model["approx_df"]
    global_mean = float(model["global_mean"])
    if user_id in approx_df.index and movie_id in approx_df.columns:
        return float(approx_df.loc[user_id, movie_id])
    return global_mean


def recommend_for_user(
    model,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend unseen movies for a user based on predicted ratings."""
    watched = set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].tolist())
    all_movies = movies_df["movieId"].tolist()
    unseen = [mid for mid in all_movies if mid not in watched]

    scored = []
    for movie_id in unseen:
        score = predict_rating(model, user_id=user_id, movie_id=int(movie_id))
        scored.append((movie_id, score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(scored, columns=["movieId", "predicted_rating"])
    result = result.merge(movies_df[["movieId", "title", "genres"]], on="movieId", how="left")
    return result[["movieId", "title", "genres", "predicted_rating"]]
