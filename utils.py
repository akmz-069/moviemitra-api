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
