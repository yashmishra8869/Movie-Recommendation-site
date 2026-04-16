"""
TMDB dataset sync utility for Indian cinema.

This script fetches popular movies from TMDB for:
- Bollywood (Hindi: hi)
- Tollywood (Telugu: te)
- Kollywood (Tamil: ta)
- Mollywood (Malayalam: ml)

It can merge with the existing local dataset or replace it.

Usage examples:
    py -3.13 tmdb_sync.py
    py -3.13 tmdb_sync.py --pages 10
    py -3.13 tmdb_sync.py --replace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "movies_sample.csv"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

LANGUAGES = {
    "Bollywood": "hi",
    "Tollywood": "te",
    "Kollywood": "ta",
    "Mollywood": "ml",
}


def read_api_key() -> str:
    """Read TMDB API key from environment or .env file."""
    env_key = os.getenv("TMDB_API_KEY", "").strip()
    if env_key:
        return env_key

    dotenv_path = BASE_DIR / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TMDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return ""


def fetch_genre_map(api_key: str) -> Dict[int, str]:
    """Fetch TMDB genre mapping for readable genre names."""
    response = requests.get(
        f"{TMDB_BASE_URL}/genre/movie/list",
        params={"api_key": api_key, "language": "en-US"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {genre["id"]: genre["name"] for genre in data.get("genres", [])}


def fetch_language_movies(
    api_key: str,
    language_code: str,
    genre_map: Dict[int, str],
    pages: int,
) -> List[Dict]:
    """Fetch popular movies for one language from TMDB discover endpoint."""
    all_rows: List[Dict] = []

    for page in range(1, pages + 1):
        response = requests.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": api_key,
                "language": "en-US",
                "with_original_language": language_code,
                "region": "IN",
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false",
                "vote_count.gte": 50,
                "page": page,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        for movie in payload.get("results", []):
            title = (movie.get("title") or "").strip()
            if not title:
                continue

            genre_ids = movie.get("genre_ids", [])
            genres = "|".join(genre_map.get(genre_id, "") for genre_id in genre_ids)
            genres = "|".join([g for g in genres.split("|") if g])

            all_rows.append(
                {
                    "title": title,
                    "genres": genres,
                    "overview": (movie.get("overview") or "").strip(),
                    "tmdb_id": movie.get("id"),
                }
            )

    return all_rows


def dedupe_by_title(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows using case-insensitive title key, keeping first occurrence."""
    if df.empty:
        return df

    dedupe_key = df["title"].astype(str).str.strip().str.lower()
    deduped = df.loc[~dedupe_key.duplicated(keep="first")].copy()
    return deduped.reset_index(drop=True)


def build_synced_dataset(pages: int, replace_existing: bool, output_path: Path) -> pd.DataFrame:
    """Build and save a merged or replaced dataset with TMDB Indian movies."""
    api_key = read_api_key()
    if not api_key or api_key == "YOUR_TMDB_API_KEY":
        raise ValueError(
            "TMDB_API_KEY is missing. Set it in environment or .env before syncing."
        )

    genre_map = fetch_genre_map(api_key)

    tmdb_rows: List[Dict] = []
    for language_name, language_code in LANGUAGES.items():
        rows = fetch_language_movies(api_key, language_code, genre_map, pages)
        tmdb_rows.extend(rows)
        print(f"Fetched {len(rows)} rows for {language_name} ({language_code}).")

    tmdb_df = pd.DataFrame(tmdb_rows, columns=["title", "genres", "overview", "tmdb_id"])
    tmdb_df = dedupe_by_title(tmdb_df)

    if replace_existing or not output_path.exists():
        final_df = tmdb_df
    else:
        existing_df = pd.read_csv(output_path)
        for col in ["title", "genres", "overview", "tmdb_id"]:
            if col not in existing_df.columns:
                existing_df[col] = ""
        existing_df = existing_df[["title", "genres", "overview", "tmdb_id"]]
        final_df = pd.concat([existing_df, tmdb_df], ignore_index=True)
        final_df = dedupe_by_title(final_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Indian movies from TMDB.")
    parser.add_argument("--pages", type=int, default=5, help="Pages per language (default: 5)")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing dataset instead of merging.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default: data/movies_sample.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.pages < 1:
        raise ValueError("--pages must be >= 1")

    final_df = build_synced_dataset(
        pages=args.pages,
        replace_existing=args.replace,
        output_path=output_path,
    )
    print(f"Saved {len(final_df)} unique movies to: {output_path}")


if __name__ == "__main__":
    main()
