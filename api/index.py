"""
Flask API for a Hybrid Movie Recommendation System.

Hybrid strategy:
1) Content-based score using TF-IDF + cosine similarity on genres + overview.
2) Collaborative score using a simulated latent-factor (SVD-like) user-movie affinity.
3) Weighted fusion of both scores using an internal default profile for demo consistency.
"""

from __future__ import annotations

from difflib import get_close_matches
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movies_sample.csv"

# Explicitly load .env so running `py api/index.py` picks up TMDB_API_KEY too.
load_dotenv(BASE_DIR / ".env")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_FALLBACK = "https://via.placeholder.com/500x750?text=No+Poster"


class HybridRecommender:
    """Hybrid recommender that combines content and collaborative signals."""

    def __init__(self, data_path: Path) -> None:
        self.movies = self._load_movies(data_path)
        self.movies["content_text"] = (
            self.movies["genres"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.movies["overview"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.content_matrix = self.vectorizer.fit_transform(self.movies["content_text"])
        self.content_similarity = cosine_similarity(self.content_matrix)

        # Simulated SVD latent factors for user-movie interaction patterns.
        rng = np.random.default_rng(seed=42)
        latent_dim = 12
        self.movie_factors = rng.normal(0, 1, (len(self.movies), latent_dim))

        self.title_to_index: Dict[str, int] = {
            title.lower(): idx for idx, title in enumerate(self.movies["title"].tolist())
        }

    @staticmethod
    def _load_movies(data_path: Path) -> pd.DataFrame:
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. Run data_loader.py to generate it."
            )

        required_columns = {"title", "genres", "overview"}
        df = pd.read_csv(data_path)
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        return df.reset_index(drop=True)

    def _get_index(self, movie_title: str) -> Optional[int]:
        normalized = movie_title.strip().lower()
        if not normalized:
            return None

        exact = self.title_to_index.get(normalized)
        if exact is not None:
            return exact

        # Allow inputs like "dangal movie" by checking if a known title is inside query.
        for title_lower, idx in self.title_to_index.items():
            if title_lower in normalized:
                return idx

        contains_matches = self.movies[
            self.movies["title"].str.lower().str.contains(normalized, na=False)
        ]
        if not contains_matches.empty:
            return int(contains_matches.index[0])

        close = get_close_matches(normalized, list(self.title_to_index.keys()), n=1, cutoff=0.6)
        if close:
            return self.title_to_index[close[0]]

        return None

    def suggest_titles(self, query: str, limit: int = 5) -> List[str]:
        suggestions = self.autocomplete(query=query, limit=limit)
        seen = {title.lower() for title in suggestions}

        close = get_close_matches(
            query.strip().lower(),
            list(self.title_to_index.keys()),
            n=limit,
            cutoff=0.5,
        )
        for title_lower in close:
            title = self.movies.iloc[self.title_to_index[title_lower]]["title"]
            if title.lower() not in seen:
                suggestions.append(title)
                seen.add(title.lower())
            if len(suggestions) >= limit:
                break

        return suggestions[:limit]

    def _simulated_user_vector(self, user_id: int) -> np.ndarray:
        """Generate a deterministic pseudo-user latent vector from user_id."""
        rng = np.random.default_rng(seed=1000 + int(user_id))
        return rng.normal(0, 1, self.movie_factors.shape[1])

    def _collaborative_scores(self, user_id: int) -> np.ndarray:
        """Compute SVD-like collaborative affinity scores for all movies."""
        user_vector = self._simulated_user_vector(user_id)
        scores = self.movie_factors @ user_vector
        # Min-max normalize for stable weighted fusion with cosine scores.
        min_s, max_s = float(scores.min()), float(scores.max())
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _content_scores(self, movie_index: int) -> np.ndarray:
        return self.content_similarity[movie_index]

    def recommend(
        self,
        movie_title: str,
        user_id: Optional[int] = None,
        top_n: int = 10,
        content_weight: float = 0.7,
        collaborative_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Return top-N recommendations for a movie query.

        Cold start handling:
        - If user_id is None/invalid, recommendation falls back to content-only mode.
        """
        movie_index = self._get_index(movie_title)
        if movie_index is None:
            return []

        content_scores = self._content_scores(movie_index)

        has_user = user_id is not None
        if has_user:
            collab_scores = self._collaborative_scores(int(user_id))
            hybrid_scores = (content_weight * content_scores) + (
                collaborative_weight * collab_scores
            )
            mode = "hybrid"
        else:
            hybrid_scores = content_scores
            mode = "content-only"

        ranked_indices = np.argsort(hybrid_scores)[::-1]

        results: List[Dict] = []
        for idx in ranked_indices:
            if idx == movie_index:
                continue
            row = self.movies.iloc[idx]
            results.append(
                {
                    "title": row["title"],
                    "genres": row.get("genres", ""),
                    "overview": row.get("overview", ""),
                    "tmdb_id": int(row["tmdb_id"]) if "tmdb_id" in row and not pd.isna(row["tmdb_id"]) else None,
                    "score": round(float(hybrid_scores[idx]), 4),
                    "mode": mode,
                }
            )
            if len(results) >= top_n:
                break

        return results

    def autocomplete(self, query: str, limit: int = 8) -> List[str]:
        if not query:
            return []
        q = query.strip().lower()
        matches = self.movies[self.movies["title"].str.lower().str.contains(q, na=False)]
        return matches["title"].head(limit).tolist()


def fetch_tmdb_poster(movie_title: str, tmdb_id: Optional[int] = None) -> str:
    """Fetch movie poster URL from TMDB, fallback safely when unavailable."""
    if not TMDB_API_KEY or TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return POSTER_FALLBACK

    try:
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            params = {"api_key": TMDB_API_KEY}
            response = requests.get(url, params=params, timeout=4)
            if response.ok:
                data = response.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"

        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {"api_key": TMDB_API_KEY, "query": movie_title}
        response = requests.get(search_url, params=search_params, timeout=4)
        if response.ok:
            results = response.json().get("results", [])
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"
    except requests.RequestException:
        pass

    return POSTER_FALLBACK


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
CORS(app)
recommender = HybridRecommender(DATA_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/autocomplete", methods=["GET"])
def api_autocomplete():
    query = request.args.get("q", "")
    suggestions = recommender.autocomplete(query=query)
    return jsonify({"suggestions": suggestions})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = request.get_json(silent=True) or {}
    movie_title = str(payload.get("movie_title", "")).strip()

    if not movie_title:
        return jsonify({"error": "movie_title is required"}), 400

    top_n = payload.get("top_n", 10)
    try:
        top_n = max(1, min(int(top_n), 20))
    except (TypeError, ValueError):
        top_n = 10

    # User profile selection is internal to keep API contract simple for UI.
    default_user_profile_id = 1

    recommendations = recommender.recommend(
        movie_title=movie_title,
        user_id=default_user_profile_id,
        top_n=top_n,
    )

    if not recommendations:
        suggestions = recommender.suggest_titles(movie_title)
        return jsonify(
            {
                "error": f"Movie '{movie_title}' not found in dataset.",
                "hint": "Run data_loader.py to regenerate sample data if needed.",
                "suggestions": suggestions,
            }
        ), 404

    for movie in recommendations:
        movie["poster_url"] = fetch_tmdb_poster(
            movie_title=movie["title"], tmdb_id=movie.get("tmdb_id")
        )

    return jsonify(
        {
            "query": movie_title,
            "total": len(recommendations),
            "recommendations": recommendations,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "movies_loaded": len(recommender.movies)})


if __name__ == "__main__":
    app.run(debug=True)
