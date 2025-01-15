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
# New Feature: Estimate Soundtrack Popularity
def estimate_soundtrack_popularity(imdb_rating, awards_text):
    soundtrack_keywords = ["music", "soundtrack", "score", "original song"]
    soundtrack_awards = any(keyword in awards_text.lower() for keyword in soundtrack_keywords)
    score = imdb_rating * 8 + (30 if soundtrack_awards else 0)
    return "High Soundtrack Popularity" if score > 80 else "Moderate Soundtrack Popularity" if score > 50 else "Low Soundtrack Popularity"

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

# Function to estimate rewatch value
def estimate_rewatch_value(imdb_rating, plot_sentiment, awards_count):
    """A movie with high IMDb rating, positive plot sentiment, and many awards is likely to have rewatch value."""
    score = (imdb_rating * 8 +
             plot_sentiment * 10 +
             awards_count * 3)
    return "High Rewatch Value" if score > 70 else "Moderate Rewatch Value" if score > 50 else "Low Rewatch Value"

# Function to estimate franchise potential
def estimate_franchise_potential(imdb_rating, genre_sentiment, awards_count):
    """Estimate the potential of a movie to become a successful franchise."""
    score = (imdb_rating * 5 +
             genre_sentiment * 10 +
             awards_count * 2)
    return "High Franchise Potential" if score > 60 else "Moderate Franchise Potential" if score > 40 else "Low Franchise Potential"

# New Feature: Estimate Cinematography Potential
def estimate_cinematography_potential(imdb_rating, awards_text):
    cinematography_keywords = ["cinematography", "visual effects", "visuals"]
    cinematography_awards = any(keyword in awards_text.lower() for keyword in cinematography_keywords)
    score = imdb_rating * 10 + (20 if cinematography_awards else 0)
    return "High Cinematography Potential" if score > 80 else "Moderate Cinematography Potential" if score > 50 else "Low Cinematography Potential"

# New Feature: Estimate Social Media Buzz
def estimate_social_media_buzz(release_year, imdb_rating, genre_sentiment):
    current_year = datetime.now().year
    recency_factor = max(0, 10 - (current_year - release_year))  # More recent movies get higher scores
    score = (imdb_rating * 5 + genre_sentiment * 5 + recency_factor * 10)
    return "High Buzz Potential" if score > 80 else "Moderate Buzz Potential" if score > 50 else "Low Buzz Potential"

# New Feature: Estimate Box Office Success
def estimate_box_office_success(imdb_rating, awards_count, genre_sentiment):
    score = (imdb_rating * 6 + awards_count * 3 + genre_sentiment * 10)
    return "High Box Office Success" if score > 75 else "Moderate Box Office Success" if score > 50 else "Low Box Office Success"

# Update process_movie_data to include the new feature
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
            plot_sentiment = TextBlob(movie_data.get("Plot", "")).sentiment.polarity if movie_data.get("Plot") else 0
            rewatch_value = estimate_rewatch_value(imdb_rating, plot_sentiment, awards_count)
            franchise_potential = estimate_franchise_potential(imdb_rating, genre_sentiment, awards_count)
            cinematography_potential = estimate_cinematography_potential(imdb_rating, movie_data.get("Awards", ""))
            
            try:
                release_year = int(movie_data.get("Year", 0))
            except ValueError:
                release_year = 0
            
            social_media_buzz = estimate_social_media_buzz(release_year, imdb_rating, genre_sentiment)
            box_office_success = estimate_box_office_success(imdb_rating, awards_count, genre_sentiment)

            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Genre Sentiment": genre_sentiment,
                "Holiday Release": holiday_release,
                "Awards Count": awards_count,
                "Streaming Platforms": ", ".join(streaming_platforms),
                "Director Popularity": director_popularity,
                "Actor Popularity": actor_popularity,
                "Rewatch Value": rewatch_value,
                "Franchise Potential": franchise_potential,
                "Cinematography Potential": cinematography_potential,
                "Social Media Buzz Potential": social_media_buzz,
                "Box Office Success": box_office_success,
            })
    
    return pd.DataFrame(data)
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
            plot_sentiment = TextBlob(movie_data.get("Plot", "")).sentiment.polarity if movie_data.get("Plot") else 0
            rewatch_value = estimate_rewatch_value(imdb_rating, plot_sentiment, awards_count)
            franchise_potential = estimate_franchise_potential(imdb_rating, genre_sentiment, awards_count)
            cinematography_potential = estimate_cinematography_potential(imdb_rating, movie_data.get("Awards", ""))
            soundtrack_popularity = estimate_soundtrack_popularity(imdb_rating, movie_data.get("Awards", ""))

            try:
                release_year = int(movie_data.get("Year", 0))
            except ValueError:
                release_year = 0
            
            social_media_buzz = estimate_social_media_buzz(release_year, imdb_rating, genre_sentiment)
            box_office_success = estimate_box_office_success(imdb_rating, awards_count, genre_sentiment)

            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Genre Sentiment": genre_sentiment,
                "Holiday Release": holiday_release,
                "Awards Count": awards_count,
                "Streaming Platforms": ", ".join(streaming_platforms),
                "Director Popularity": director_popularity,
                "Actor Popularity": actor_popularity,
                "Rewatch Value": rewatch_value,
                "Franchise Potential": franchise_potential,
                "Cinematography Potential": cinematography_potential,
                "Soundtrack Popularity": soundtrack_popularity,
                "Social Media Buzz Potential": social_media_buzz,
                "Box Office Success": box_office_success,
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
