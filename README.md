# Hybrid Movie Recommendation System

A complete B.Tech Final Year Project using a Hybrid Recommender approach:
- Content-Based Filtering with TF-IDF + Cosine Similarity
- Simulated Collaborative Filtering with SVD-style latent factors
- Flask backend (Vercel-ready)
- Tailwind CSS dark-themed frontend with autocomplete-like search
- TMDB poster integration
- Multi-industry dataset support (Bollywood, Tollywood, Kollywood, Mollywood)

## Project Structure

- api/index.py: Flask app and hybrid recommendation engine
- templates/index.html: Netflix-style frontend
- data_loader.py: Generates curated demo dataset
- tmdb_sync.py: Pulls larger Indian movie catalog from TMDB by language
- data/movies_sample.csv: Pre-generated dataset for instant run
- requirements.txt: Python dependencies
- vercel.json: Vercel routing/build config for Python runtime

## Hybrid Logic (Viva Summary)

The recommender combines two scores for each candidate movie:

1. Content Score
- Build text features from genres + overview.
- Apply TF-IDF vectorization.
- Compute cosine similarity against the selected movie.

2. Collaborative Score (Simulated)
- Use deterministic pseudo user latent vectors from an internal default profile.
- Dot-product with movie latent factors (SVD-like behavior).
- Min-max normalize score to blend with cosine similarity.

3. Final Hybrid Score
- hybrid_score = (0.7 * content_score) + (0.3 * collaborative_score)

Personalization Handling:
- UI does not request user_id; backend uses a stable internal profile.

## Local Setup (Windows)

Use Python 3.13 via py launcher:

1. Install dependencies (standard)

```powershell
Set-Location "D:\Buisness\Fleet_Manager-main\ML Project"
py -3.13 -m pip install -r requirements.txt
```

Alternative for reproducible installs (pinned versions):

```powershell
py -3.13 -m pip install -r requirements.lock.txt
```

2. Configure environment variable

```powershell
Copy-Item .env.example .env
# Edit .env and set TMDB_API_KEY
```

3. (Optional) Regenerate sample dataset

```powershell
py -3.13 data_loader.py
```

4. (Optional) Pull more Indian movies from TMDB

```powershell
# Uses TMDB_API_KEY from .env or environment
py -3.13 tmdb_sync.py --pages 10
```

Use --replace to rebuild dataset only from TMDB results:

```powershell
py -3.13 tmdb_sync.py --pages 10 --replace
```

5. Run application

```powershell
py -3.13 api/index.py
```

6. Open in browser
- http://127.0.0.1:5000

## API Endpoints

- GET /health
  - Returns API health status and loaded movie count.

- GET /api/autocomplete?q=<query>
  - Returns title suggestions for search box.

- POST /api/recommend
  - Request body:

```json
{
  "movie_title": "Inception",
  "top_n": 10
}
```

## Vercel Deployment

1. Push project to GitHub.
2. Import repository in Vercel.
3. Add the TMDB key in Vercel Project Settings:
  - Open the project in Vercel.
  - Go to Settings -> Environment Variables.
  - Add `TMDB_API_KEY` with your real TMDB v3 API key.
  - Scope it to Production, Preview, and Development if you want it available everywhere.
4. Deploy.

Vercel reads:
- vercel.json for route/build setup
- api/index.py as Python server entrypoint

Local development reads:
- `.env` for `TMDB_API_KEY`
- `.env` is ignored by git, so the key stays out of the repository.

## Notes

- If python command does not work on Windows, use py -3.13.
- The app works immediately with the included data/movies_sample.csv file.
- Viva deck structure is available in VIVA_OUTLINE.md.
