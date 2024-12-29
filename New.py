import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import re
import random

# Function to fetch movie data from OMDb API
def fetch_movie_data(title, api_key="your_api_key_here"):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to analyze genre sentiment
def analyze_genre_sentiment(genres):
    sentiment_scores = []
    for genre in genres.split(","):
        analysis = TextBlob(genre.strip())
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Function to check if the release date is a holiday
def is_holiday_release(release_date):
    try:
        release_date = datetime.strptime(release_date, "%d %b %Y")
        return release_date.month in [11, 12]  # Holiday season: November and December
    except ValueError:
        return False

# Function to extract the number of awards won
def extract_awards_count(awards_text):
    if not awards_text:
        return 0
    try:
        match = re.search(r"(\d+) win", awards_text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0

# Function to simulate streaming platform availability
def simulate_streaming_availability(title):
    platforms = ["Netflix", "Amazon Prime", "Hulu", "Disney+"]
    available_platforms = random.sample(platforms, random.randint(1, len(platforms)))
    return available_platforms

# Function to estimate director's popularity
def get_director_popularity(director, api_key):
    if not director:
        return 0.0
    search_url = f"http://www.omdbapi.com/?s={director}&type=movie&apikey={api_key}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return 0.0

    search_results = response.json()
    if search_results.get("Response") != "True":
        return 0.0

    total_rating = 0.0
    movie_count = 0
    for movie in search_results.get("Search", []):
        movie_data = fetch_movie_data(movie.get("Title"), api_key)
        if movie_data and movie_data.get("Response") == "True" and movie_data.get("imdbRating"):
            total_rating += float(movie_data.get("imdbRating"))
            movie_count += 1

    return total_rating / movie_count if movie_count > 0 else 0.0

# Function to estimate actor popularity
def get_actor_popularity(actor, api_key):
    if not actor:
        return 0.0
    search_url = f"http://www.omdbapi.com/?s={actor}&type=movie&apikey={api_key}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return 0.0

    search_results = response.json()
    if search_results.get("Response") != "True":
        return 0.0

    total_rating = 0.0
    movie_count = 0
    for movie in search_results.get("Search", []):
        movie_data = fetch_movie_data(movie.get("Title"), api_key)
        if movie_data and movie_data.get("Response") == "True" and movie_data.get("imdbRating"):
            total_rating += float(movie_data.get("imdbRating"))
            movie_count += 1

    return total_rating / movie_count if movie_count > 0 else 0.0

# New feature: Predict box office success
def predict_box_office_success(imdb_rating, director_popularity, actor_popularity, holiday_release, genre_sentiment):
    score = (imdb_rating * 10 + 
             director_popularity * 5 + 
             actor_popularity * 5 + 
             (10 if holiday_release else 0) + 
             genre_sentiment * 20)
    return "Box Office Hit" if score > 80 else "Likely Average"

def estimate_franchise_potential(genre, awards_count, imdb_rating, director_popularity, actor_popularity):
    franchise_genres = ["Action", "Adventure", "Sci-Fi", "Fantasy"]
    is_franchise_genre = any(g.strip() in franchise_genres for g in genre.split(","))
    
    score = (
        (10 if is_franchise_genre else 0) +
        awards_count * 2 +
        imdb_rating * 10 +
        director_popularity * 5 +
        actor_popularity * 5
    )
    return "High Potential" if score > 70 else "Low Potential"


# New feature: Estimate international appeal
def estimate_international_appeal(genre, imdb_rating, director_popularity, awards_count):
    international_genres = ["Drama", "Action", "Adventure", "Sci-Fi", "Fantasy", "Animation"]
    is_international_genre = any(g.strip() in international_genres for g in genre.split(","))
    
    score = (
        (15 if is_international_genre else 0) +
        imdb_rating * 8 +
        director_popularity * 6 +
        awards_count * 3
    )
    return "High Appeal" if score > 75 else "Moderate Appeal" if score > 50 else "Low Appeal"

# Update process_movie_data to include international appeal
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            genre_sentiment = analyze_genre_sentiment(movie_data.get("Genre", ""))
            holiday_release = is_holiday_release(movie_data.get("Released", ""))
            awards_count = extract_awards_count(movie_data.get("Awards", ""))
            streaming_platforms = simulate_streaming_availability(title)
            director_popularity = get_director_popularity(movie_data.get("Director", ""), api_key)
            lead_actor = movie_data.get("Actors", "").split(",")[0] if movie_data.get("Actors") else ""
            actor_popularity = get_actor_popularity(lead_actor, api_key)
            imdb_rating = float(movie_data.get("imdbRating", 0))
            box_office_prediction = predict_box_office_success(imdb_rating, director_popularity, actor_popularity, holiday_release, genre_sentiment)
            franchise_potential = estimate_franchise_potential(movie_data.get("Genre", ""), awards_count, imdb_rating, director_popularity, actor_popularity)
            international_appeal = estimate_international_appeal(movie_data.get("Genre", ""), imdb_rating, director_popularity, awards_count)

            data.append({
                "Title": movie_data.get("Title"),
                "Year": movie_data.get("Year"),
                "IMDb Rating": imdb_rating,
                "Genre Sentiment": genre_sentiment,
                "Holiday Release": holiday_release,
                "Awards Count": awards_count,
                "Streaming Platforms": ", ".join(streaming_platforms),
                "Director Popularity": director_popularity,
                "Actor Popularity": actor_popularity,
                "Box Office Prediction": box_office_prediction,
                "Franchise Potential": franchise_potential,
                "International Appeal": international_appeal,  # New feature
            })
    
    return pd.DataFrame(data)

# Example usage
def main():
    api_key = "your_api_key_here"
    movie_titles = ["Inception", "The Dark Knight", "Interstellar"]
    result_df = process_movie_data(movie_titles, api_key)
    print(result_df)

if __name__ == "__main__":
    main()
