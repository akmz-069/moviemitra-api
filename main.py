from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import pickle
import pandas as pd
from utils import fetch_poster_and_overview

# ===============================
# Load dataset and similarity matrix
# ===============================
movies_dict = pickle.load(open("models/movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open("models/similarity.pkl", "rb"))

# In-memory user watchlists
watchlists: Dict[str, List[str]] = {}

# ===============================
# FastAPI Initialization
# ===============================
app = FastAPI(
    title="🎬 MovieMitra API",
    version="3.0",
    description="MovieMitra API for movie recommendations, popular movies, and watchlist management.",
)

# ===============================
# Data Models
# ===============================
class Movie(BaseModel):
    movie_id: int
    title: str
    overview: Optional[str] = ""
    poster_url: Optional[str] = ""

class WatchlistItem(BaseModel):
    username: str
    movie_title: str

# ===============================
# Root Endpoint
# ===============================
@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Welcome to MovieMitra API 🎬",
        "available_endpoints": [
            "/movies",
            "/movies/popular",
            "/movies/dropdown",
            "/movies/{movie_id}",
            "/movies/title/{movie_title}",
            "/recommend",
            "/recommend/title/{movie_title}",
            "/watchlist/{username}",
            "/watchlist/add",
            "/watchlist/remove",
        ],
    }

# ===============================
# 🎬 Dropdown Movie Names Endpoint
# ===============================
@app.get("/movies/dropdown")
def get_dropdown_movies(
    movie_id: Optional[int] = Query(None, description="Optional movie ID to get its title"),
    movie_title: Optional[str] = Query(None, description="Optional movie title to get its ID")
):
    """
    Returns:
    - All movie titles (for dropdowns/autocomplete)
    - A specific movie title by ID
    - A movie ID by title
    """
    try:
        if movie_id is not None:
            movie_row = movies[movies["movie_id"] == movie_id]
            if movie_row.empty:
                raise HTTPException(status_code=404, detail=f"Movie ID '{movie_id}' not found")
            return {"movie_id": movie_id, "title": movie_row.iloc[0]["title"]}

        if movie_title is not None:
            movie_row = movies[movies["title"].str.lower() == movie_title.lower()]
            if movie_row.empty:
                raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
            return {"movie_id": int(movie_row.iloc[0]['movie_id']), "title": movie_title}

        movie_names = movies["title"].dropna().tolist()
        return {"movies": movie_names}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# All Movies Endpoint
# ===============================
@app.get("/movies", response_model=List[Movie])
def get_all_movies(limit: int = Query(50, description="Number of movies to fetch")):
    """
    Fetch all movies up to a given limit.
    """
    result = []
    for _, row in movies.head(limit).iterrows():
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        result.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return result

# ===============================
# 🆕 Popular Movies Endpoint (fetch 40 movies)
# ===============================
@app.get("/movies/popular", response_model=List[Movie])
def get_popular_movies(limit: int = Query(40, description="Number of popular movies to fetch")):
    """
    Returns top popular movies sorted by vote_count or popularity.
    """
    try:
        if "vote_count" in movies.columns:
            popular_movies = movies.sort_values(by="vote_count", ascending=False).head(limit)
        elif "popularity" in movies.columns:
            popular_movies = movies.sort_values(by="popularity", ascending=False).head(limit)
        else:
            popular_movies = movies.head(limit)

        results = []
        for _, row in popular_movies.iterrows():
            poster, overview, _ = fetch_poster_and_overview(row.movie_id)
            results.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===============================
# Get Movie by ID
# ===============================
@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie_by_id(movie_id: int):
    movie_row = movies[movies["movie_id"] == movie_id]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    row = movie_row.iloc[0]
    poster, overview, _ = fetch_poster_and_overview(row.movie_id)
    return Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster)

# ===============================
# Get Movie by Title
# ===============================
@app.get("/movies/title/{movie_title}", response_model=Movie)
def get_movie_by_title(movie_title: str):
    movie_row = movies[movies["title"].str.lower() == movie_title.lower()]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
    row = movie_row.iloc[0]
    poster, overview, _ = fetch_poster_and_overview(row.movie_id)
    return Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster)

# ===============================
# Unified Recommendation Endpoint
# ===============================
@app.get("/recommend", response_model=List[Movie])
def recommend(
    movie_id: Optional[int] = Query(None, description="Movie ID"),
    movie_title: Optional[str] = Query(None, description="Movie title"),
):
    """
    Generate recommendations based on a movie ID or title.
    """
    if not movie_id and not movie_title:
        raise HTTPException(status_code=400, detail="Provide either 'movie_id' or 'movie_title'")

    if movie_title:
        movie_row = movies[movies["title"].str.lower() == movie_title.lower()]
        if movie_row.empty:
            raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")
        movie_index = movie_row.index[0]
    else:
        movie_index = movies[movies["movie_id"] == movie_id].index
        if movie_index.empty:
            raise HTTPException(status_code=404, detail=f"Movie ID '{movie_id}' not found")
        movie_index = movie_index[0]

    distances = similarity[movie_index]
    similar_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    results = []
    for i in similar_movies:
        row = movies.iloc[i[0]]
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        results.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return results

# ===============================
# Recommendation by Title Only
# ===============================
@app.get("/recommend/title/{movie_title}", response_model=List[Movie])
def get_recommendations_by_title(movie_title: str):
    """
    Get recommendations based on a specific movie title.
    """
    movie_row = movies[movies["title"].str.lower() == movie_title.lower()]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found")

    movie_index = movie_row.index[0]
    distances = similarity[movie_index]
    similar_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]

    recommendations = []
    for i in similar_movies:
        row = movies.iloc[i[0]]
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        recommendations.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return recommendations

# ===============================
# Watchlist Management
# ===============================
@app.get("/watchlist/{username}", response_model=List[Movie])
def get_watchlist(username: str):
    """
    Fetch a user's watchlist.
    """
    movie_titles = watchlists.get(username, [])
    result = []
    for title in movie_titles:
        row = movies[movies["title"] == title]
        if row.empty:
            continue
        row = row.iloc[0]
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        result.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return result

@app.post("/watchlist/add")
def add_to_watchlist(item: WatchlistItem):
    watchlists.setdefault(item.username, [])
    if item.movie_title not in watchlists[item.username]:
        watchlists[item.username].append(item.movie_title)
        return {"status": "success", "message": f"✅ {item.movie_title} added to {item.username}'s watchlist"}
    else:
        return {"status": "info", "message": f"ℹ️ {item.movie_title} already in watchlist"}

@app.post("/watchlist/remove")
def remove_from_watchlist(item: WatchlistItem):
    if item.username in watchlists and item.movie_title in watchlists[item.username]:
        watchlists[item.username].remove(item.movie_title)
        return {"status": "success", "message": f"❌ {item.movie_title} removed from {item.username}'s watchlist"}
    else:
        raise HTTPException(status_code=404, detail=f"{item.movie_title} not found in {item.username}'s watchlist")
