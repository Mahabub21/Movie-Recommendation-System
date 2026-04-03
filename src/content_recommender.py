import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentRecommender:
    def __init__(self, min_df: int = 1) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df)
        self.tfidf_matrix = None
        self.movies = None
        self.title_to_index: dict[str, int] = {}

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", str(title).strip().lower())

    @staticmethod
    def _tokenize_title(title: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9 ]", " ", str(title).lower())
        tokens = [t for t in cleaned.split() if t]
        return set(tokens)

    def fit(self, movies_df: pd.DataFrame) -> "ContentRecommender":
        self.movies = movies_df.reset_index(drop=True).copy()
        self.movies["content"] = self.movies["content"].fillna("")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["content"])
        self.title_to_index = {
            self._normalize_title(title): idx
            for idx, title in enumerate(self.movies["title"].tolist())
        }
        return self

    def _resolve_title(self, title: str) -> int | None:
        if self.movies is None:
            return None

        normalized = self._normalize_title(title)
        if normalized in self.title_to_index:
            return self.title_to_index[normalized]

        for movie_title, idx in self.title_to_index.items():
            if normalized in movie_title:
                return idx

        query_tokens = self._tokenize_title(normalized)
        if not query_tokens:
            return None

        # Match titles that contain all query tokens, allowing formats like "Matrix, The (1999)".
        for idx, movie_title in enumerate(self.movies["title"].tolist()):
            movie_tokens = self._tokenize_title(movie_title)
            if query_tokens.issubset(movie_tokens):
                return idx
        return None

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        if self.movies is None or self.tfidf_matrix is None:
            raise ValueError("Model is not fitted.")

        idx = self._resolve_title(title)
        if idx is None:
            return pd.DataFrame(columns=["title", "similarity"])

        similarities = linear_kernel(self.tfidf_matrix[idx : idx + 1], self.tfidf_matrix).flatten()
        scored = list(enumerate(similarities))
        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        scored = [(i, s) for i, s in scored if i != idx][:top_n]

        rec_indices = [i for i, _ in scored]
        rec_scores = [float(s) for _, s in scored]

        result = self.movies.iloc[rec_indices][["movieId", "title", "genres"]].copy()
        result["similarity"] = rec_scores
        return result.reset_index(drop=True)
