import requests

def fetch_poster_and_overview(movie_id):
    try:
        response = requests.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=eefa1a436c5402278de86eff4026185c&language=en-US'
        )
        data = response.json()
        title = data.get('title', 'Unknown')
        poster_url = (
            "https://image.tmdb.org/t/p/w500/" + data['poster_path']
            if data.get('poster_path')
            else "https://via.placeholder.com/500x750?text=No+Image"
        )
        overview = data.get('overview', 'No description available.')
        return poster_url, overview, title
    except:
        return "https://via.placeholder.com/500x750?text=No+Image", "No description available.", "Unknown"

def fetch_tmdb_movie_data(movie_id):
    """
    Fetch complete TMDB movie data for a given movie ID.
    Returns a dictionary matching the TMDBMovie model structure.
    """
    try:
        response = requests.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=eefa1a436c5402278de86eff4026185c&language=en-US'
        )
        data = response.json()
        
        return {
            'adult': data.get('adult', False),
            'backdrop_path': data.get('backdrop_path'),
            'genre_ids': data.get('genre_ids', []),
            'id': data.get('id', movie_id),
            'original_language': data.get('original_language', 'en'),
            'original_title': data.get('original_title', ''),
            'overview': data.get('overview'),
            'popularity': data.get('popularity', 0.0),
            'poster_path': data.get('poster_path'),
            'release_date': data.get('release_date'),
            'title': data.get('title', 'Unknown'),
            'video': data.get('video', False),
            'vote_average': data.get('vote_average', 0.0),
            'vote_count': data.get('vote_count', 0),
            'isFavourite': None
        }
    except Exception as e:
        # Return default values if API call fails
        return {
            'adult': False,
            'backdrop_path': None,
            'genre_ids': [],
            'id': movie_id,
            'original_language': 'en',
            'original_title': 'Unknown',
            'overview': 'No description available.',
            'popularity': 0.0,
            'poster_path': None,
            'release_date': None,
            'title': 'Unknown',
            'video': False,
            'vote_average': 0.0,
            'vote_count': 0,
            'isFavourite': None
        }