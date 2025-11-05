from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import pandas as pd
from utils import fetch_poster_and_overview

movies_dict = pickle.load(open(".venv/movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open("models/similarity.pkl", "rb"))

watchlists = {}

app = FastAPI(title="MovieMitra API", version="1.0")

class Movie(BaseModel):
    movie_id: int
    title: str
    overview: Optional[str] = ""
    poster_url: Optional[str] = ""

class WatchlistItem(BaseModel):
    username: str
    movie_title: str

@app.get("/")
def read_root():
    return {"message": "Welcome to MovieMitra API"}

@app.get("/movies", response_model=List[Movie])
def get_all_movies(limit: int = 50):
    result = []
    for _, row in movies.head(limit).iterrows():
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        result.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return result

@app.get("/movies/{movie_id}", response_model=Movie)
def get_movie(movie_id: int):
    movie_row = movies[movies["movie_id"] == movie_id]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    row = movie_row.iloc[0]
    poster, overview, _ = fetch_poster_and_overview(row.movie_id)
    return Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster)

@app.get("/recommend/{movie_title}", response_model=List[Movie])
def get_recommendations(movie_title: str):
    if movie_title not in movies["title"].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    movie_index = movies[movies["title"] == movie_title].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    result = []
    for i in movies_list:
        row = movies.iloc[i[0]]
        poster, overview, _ = fetch_poster_and_overview(row.movie_id)
        result.append(Movie(movie_id=row.movie_id, title=row.title, overview=overview, poster_url=poster))
    return result

@app.get("/watchlist/{username}", response_model=List[Movie])
def get_watchlist(username: str):
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

@app.post("/watchlist/add", response_model=dict)
def add_to_watchlist(item: WatchlistItem):
    watchlists.setdefault(item.username, [])
    if item.movie_title not in watchlists[item.username]:
        watchlists[item.username].append(item.movie_title)
    return {"message": f"{item.movie_title} added to {item.username}'s watchlist"}

@app.post("/watchlist/remove", response_model=dict)
def remove_from_watchlist(item: WatchlistItem):
    if item.username in watchlists and item.movie_title in watchlists[item.username]:
        watchlists[item.username].remove(item.movie_title)
        return {"message": f"{item.movie_title} removed from {item.username}'s watchlist"}
    else:
        raise HTTPException(status_code=404, detail="Movie not in watchlist")
