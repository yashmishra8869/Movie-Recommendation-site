"""Vercel-safe Flask API for Hybrid Movie Recommendation."""

from __future__ import annotations

from difflib import get_close_matches
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_PATH = os.path.join(os.path.dirname(__file__), "movies.csv")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
POSTER_FALLBACK = "https://via.placeholder.com/500x750?text=No+Poster"


def _normalize_scores(scores) -> List[float]:
    """Normalize scores to [0, 1] for stable weighted fusion."""
    if scores is None or len(scores) == 0:
        return []

    min_s = float(scores.min())
    max_s = float(scores.max())
    if abs(max_s - min_s) < 1e-9:
        return [0.0 for _ in range(len(scores))]
    return [float((value - min_s) / (max_s - min_s)) for value in scores]


class HybridEngine:
    """Hybrid recommender combining TF-IDF similarity and simulated SVD signal."""

    def __init__(self, csv_path: str) -> None:
        self.movies = self._load_movies(csv_path)
        self.movies["content_text"] = (
            self.movies["genres"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.movies["overview"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.content_matrix = self.vectorizer.fit_transform(self.movies["content_text"])
        self.content_similarity = cosine_similarity(self.content_matrix)

        self.title_to_index: Dict[str, int] = {
            title.lower(): idx for idx, title in enumerate(self.movies["title"].tolist())
        }

        self.default_user_index = 0
        self.user_latent = None
        self.movie_latent = None
        self._build_collaborative_projection()

    @staticmethod
    def _load_movies(csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Movie dataset not found. Please ensure api/movies.csv exists in deployment."
            )

        df = pd.read_csv(csv_path)
        required_columns = {"title", "genres", "overview"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"movies.csv is missing required columns: {sorted(missing)}")

        df["title"] = df["title"].fillna("").astype(str)
        df["genres"] = df["genres"].fillna("").astype(str)
        df["overview"] = df["overview"].fillna("").astype(str)
        return df.reset_index(drop=True)

    def _build_collaborative_projection(self) -> None:
        """Build simulated user-item matrix and project with SVD."""
        genre_matrix = self.movies["genres"].str.get_dummies(sep="|")
        if genre_matrix.empty:
            genre_matrix = pd.DataFrame({"Unknown": [1] * len(self.movies)})

        genre_cols = list(genre_matrix.columns)
        simulated_profiles = [
            {"Action": 1.0, "Adventure": 0.8, "Sci-Fi": 0.7, "Thriller": 0.6},
            {"Drama": 1.0, "Romance": 0.7, "Family": 0.5, "History": 0.5},
            {"Crime": 1.0, "Mystery": 0.8, "Thriller": 0.8, "Drama": 0.6},
            {"Comedy": 1.0, "Adventure": 0.6, "Fantasy": 0.5, "Romance": 0.4},
            {"War": 1.0, "History": 0.8, "Action": 0.6, "Biography": 0.5},
            {"Sport": 1.0, "Drama": 0.7, "Action": 0.4, "Family": 0.4},
        ]

        profile_rows = []
        for profile in simulated_profiles:
            row = []
            for col in genre_cols:
                row.append(profile.get(col, 0.15))
            profile_rows.append(row)

        profile_df = pd.DataFrame(profile_rows, columns=genre_cols)
        ratings = profile_df.values @ genre_matrix.T.values

        min_dimension = min(ratings.shape[0], ratings.shape[1])
        if min_dimension <= 1:
            return

        n_components = min(8, min_dimension - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_latent = svd.fit_transform(ratings)
        self.movie_latent = svd.components_.T

    def _resolve_index(self, movie_title: str) -> Optional[int]:
        query = movie_title.strip().lower()
        if not query:
            return None

        exact = self.title_to_index.get(query)
        if exact is not None:
            return exact

        contains = self.movies[self.movies["title"].str.lower().str.contains(query, na=False)]
        if not contains.empty:
            return int(contains.index[0])

        close = get_close_matches(query, list(self.title_to_index.keys()), n=1, cutoff=0.6)
        if close:
            return self.title_to_index[close[0]]

        return None

    def autocomplete(self, query: str, limit: int = 8) -> List[str]:
        if not query:
            return []
        matches = self.movies[
            self.movies["title"].str.lower().str.contains(query.strip().lower(), na=False)
        ]
        return matches["title"].head(limit).tolist()

    def suggest_titles(self, query: str, limit: int = 5) -> List[str]:
        suggestions = self.autocomplete(query=query, limit=limit)
        close = get_close_matches(
            query.strip().lower(),
            list(self.title_to_index.keys()),
            n=limit,
            cutoff=0.5,
        )
        for title_lower in close:
            title = self.movies.iloc[self.title_to_index[title_lower]]["title"]
            if title not in suggestions:
                suggestions.append(title)
            if len(suggestions) >= limit:
                break
        return suggestions[:limit]

    def _collaborative_scores(self) -> List[float]:
        if self.user_latent is None or self.movie_latent is None:
            return [0.0 for _ in range(len(self.movies))]

        user_vector = self.user_latent[self.default_user_index].reshape(1, -1)
        raw_scores = cosine_similarity(user_vector, self.movie_latent).flatten()
        return _normalize_scores(raw_scores)

    def recommend(
        self,
        movie_title: str,
        top_n: int = 10,
        content_weight: float = 0.75,
        collaborative_weight: float = 0.25,
    ) -> Tuple[List[Dict], List[str]]:
        movie_index = self._resolve_index(movie_title)
        if movie_index is None:
            return [], self.suggest_titles(movie_title)

        content_scores = _normalize_scores(self.content_similarity[movie_index])
        collab_scores = self._collaborative_scores()

        hybrid_scores = [
            (content_weight * content_scores[i]) + (collaborative_weight * collab_scores[i])
            for i in range(len(self.movies))
        ]

        ranked = sorted(
            [(idx, score) for idx, score in enumerate(hybrid_scores) if idx != movie_index],
            key=lambda item: item[1],
            reverse=True,
        )

        results: List[Dict] = []
        for idx, score in ranked[:top_n]:
            row = self.movies.iloc[idx]
            tmdb_id = None
            if "tmdb_id" in self.movies.columns and not pd.isna(row.get("tmdb_id")):
                try:
                    tmdb_id = int(float(row["tmdb_id"]))
                except (TypeError, ValueError):
                    tmdb_id = None

            results.append(
                {
                    "title": row.get("title", ""),
                    "genres": row.get("genres", ""),
                    "overview": row.get("overview", ""),
                    "tmdb_id": tmdb_id,
                    "score": round(float(score), 4),
                    "mode": "hybrid",
                }
            )

        return results, []


def fetch_tmdb_poster(movie_title: str, tmdb_id: Optional[int]) -> str:
    """Fetch TMDB poster URL and fail gracefully on API/network errors."""
    if not TMDB_API_KEY:
        return POSTER_FALLBACK

    try:
        if tmdb_id is not None:
            by_id = requests.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={"api_key": TMDB_API_KEY},
                timeout=5,
            )
            if by_id.ok:
                poster_path = by_id.json().get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"

        search = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": TMDB_API_KEY, "query": movie_title},
            timeout=5,
        )
        if search.ok:
            results = search.json().get("results", [])
            if results and results[0].get("poster_path"):
                return f"{TMDB_IMAGE_BASE}{results[0]['poster_path']}"
    except requests.RequestException:
        return POSTER_FALLBACK

    return POSTER_FALLBACK


app = Flask(__name__, template_folder="../templates")
CORS(app)

ENGINE: Optional[HybridEngine] = None
STARTUP_ERROR: Optional[str] = None


def get_engine() -> Optional[HybridEngine]:
    """Initialize engine lazily to avoid crashing serverless cold starts."""
    global ENGINE, STARTUP_ERROR
    if ENGINE is not None:
        return ENGINE
    if STARTUP_ERROR is not None:
        return None

    try:
        ENGINE = HybridEngine(CSV_PATH)
        return ENGINE
    except Exception as exc:  # pragma: no cover - defensive path for deployment
        STARTUP_ERROR = str(exc)
        return None


@app.route("/", methods=["GET"])
def home():
    engine = get_engine()
    if engine is None:
        return (
            "<h2>Hybrid Movie Recommendation System</h2>"
            "<p>Service is temporarily unavailable.</p>"
            f"<p><strong>Reason:</strong> {STARTUP_ERROR}</p>",
            503,
        )

    try:
        return render_template("index.html")
    except Exception:
        return (
            "<h2>Hybrid Movie Recommendation System API</h2>"
            "<p>Frontend template not found, but API is running.</p>",
            200,
        )


@app.route("/api/autocomplete", methods=["GET"])
def api_autocomplete():
    engine = get_engine()
    if engine is None:
        return jsonify({"suggestions": [], "message": "Service unavailable."}), 503

    query = request.args.get("q", "")
    return jsonify({"suggestions": engine.autocomplete(query)})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    engine = get_engine()
    if engine is None:
        return (
            jsonify(
                {
                    "error": "Recommendation service unavailable.",
                    "message": STARTUP_ERROR,
                }
            ),
            503,
        )

    payload = request.get_json(silent=True) or {}
    movie_title = str(payload.get("movie_title", "")).strip()
    if not movie_title:
        return jsonify({"error": "movie_title is required."}), 400

    try:
        top_n = max(1, min(int(payload.get("top_n", 10)), 20))
    except (TypeError, ValueError):
        top_n = 10

    recommendations, suggestions = engine.recommend(movie_title=movie_title, top_n=top_n)
    if not recommendations:
        return (
            jsonify(
                {
                    "error": f"Movie '{movie_title}' not found in dataset.",
                    "suggestions": suggestions,
                }
            ),
            404,
        )

    for movie in recommendations:
        movie["poster_url"] = fetch_tmdb_poster(movie["title"], movie.get("tmdb_id"))

    return jsonify(
        {
            "query": movie_title,
            "total": len(recommendations),
            "recommendations": recommendations,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    engine = get_engine()
    if engine is None:
        return jsonify({"status": "degraded", "movies_loaded": 0, "message": STARTUP_ERROR}), 503
    return jsonify({"status": "ok", "movies_loaded": len(engine.movies)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
