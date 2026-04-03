from pathlib import Path

import pandas as pd


def load_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load required CSV files from the dataset directory."""
    data_path = Path(data_dir)
    datasets = {
        "movies": pd.read_csv(data_path / "movies.csv"),
        "ratings": pd.read_csv(data_path / "ratings.csv"),
        "tags": pd.read_csv(data_path / "tags.csv"),
        "links": pd.read_csv(data_path / "links.csv"),
    }
    return datasets


def clean_and_prepare_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Clean data, merge tags with genres, and build content features."""
    datasets = load_datasets(data_dir)
    movies = datasets["movies"].copy()
    ratings = datasets["ratings"].copy()
    tags = datasets["tags"].copy()
    links = datasets["links"].copy()

    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])

    tags = tags.dropna(subset=["movieId", "tag"]).copy()
    tags["movieId"] = tags["movieId"].astype(int)
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()

    links = links.dropna(subset=["movieId"]).copy()
    links["movieId"] = links["movieId"].astype(int)

    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(sorted(set(t for t in x if t))))
        .reset_index(name="tags_text")
    )

    movies = movies.merge(tags_grouped, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")
    movies["genres_text"] = movies["genres"].str.replace("|", " ", regex=False)
    movies["content"] = (movies["genres_text"] + " " + movies["tags_text"]).str.strip()

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
    }


def get_popular_movies(movies: pd.DataFrame, ratings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top movies by number of ratings."""
    rating_counts = ratings.groupby("movieId").size().reset_index(name="rating_count")
    popular = movies[["movieId", "title"]].merge(rating_counts, on="movieId", how="inner")
    popular = popular.sort_values("rating_count", ascending=False).head(top_n)
    return popular


def get_top_rated_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return highest rated movies with a minimum number of ratings."""
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    filtered = stats[stats["count"] >= min_ratings]
    top_rated = movies[["movieId", "title"]].merge(filtered, on="movieId", how="inner")
    top_rated = top_rated.sort_values("mean", ascending=False).head(top_n)
    return top_rated


def get_top_genres(movies: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return top genre frequencies for plotting."""
    return movies["genres"].str.split("|").explode().value_counts().head(top_n)
