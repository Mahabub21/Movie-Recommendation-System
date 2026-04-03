from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rating_distribution(ratings: pd.DataFrame, out_path: str | Path) -> None:
    """Plot and save rating histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(ratings["rating"], bins=10, color="#3a6ea5", edgecolor="black")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_top_genres(movies: pd.DataFrame, out_path: str | Path, top_n: int = 10) -> None:
    """Plot and save top genres bar chart."""
    genre_counts = movies["genres"].str.split("|").explode().value_counts().head(top_n)

    plt.figure(figsize=(10, 5))
    genre_counts.plot(kind="bar", color="#4caf50")
    plt.title("Top Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_user_movie_heatmap(
    ratings: pd.DataFrame,
    out_path: str | Path,
    max_users: int = 50,
    max_movies: int = 50,
) -> None:
    """Optional heatmap for a subset of user-movie matrix."""
    subset = ratings.copy()
    top_users = subset["userId"].value_counts().head(max_users).index
    subset = subset[subset["userId"].isin(top_users)]

    top_movies = subset["movieId"].value_counts().head(max_movies).index
    subset = subset[subset["movieId"].isin(top_movies)]

    matrix = subset.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0)

    plt.figure(figsize=(12, 7))
    sns.heatmap(matrix, cmap="YlGnBu")
    plt.title("User-Movie Rating Matrix (Sample)")
    plt.xlabel("Movie ID")
    plt.ylabel("User ID")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
