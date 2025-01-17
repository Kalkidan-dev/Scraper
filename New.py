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

# Function to extract awards count
def extract_awards_count(awards_text):
    if not awards_text:
        return 0
    try:
        match = re.search(r"(\d+) win", awards_text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0

# Function to calculate Diversity Index for cast
def calculate_diversity_index(cast_list):
    if not cast_list:
        return 0.0

    # Dummy dataset for demonstration
    cast_details = {
        "Leonardo DiCaprio": {"ethnicity": "Caucasian", "gender": "Male", "nationality": "USA"},
        "Morgan Freeman": {"ethnicity": "African-American", "gender": "Male", "nationality": "USA"},
        "Marion Cotillard": {"ethnicity": "Caucasian", "gender": "Female", "nationality": "France"},
        # Add more actors for real calculations
    }

    ethnicities = set()
    genders = set()
    nationalities = set()

    for actor in cast_list:
        details = cast_details.get(actor.strip())
        if details:
            ethnicities.add(details["ethnicity"])
            genders.add(details["gender"])
            nationalities.add(details["nationality"])

    diversity_score = (
        len(ethnicities) + len(genders) + len(nationalities)
    ) / (3 * len(cast_list))

    return round(diversity_score * 100, 2)

# Function to estimate climate suitability
def estimate_climate_suitability(release_date, genre):
    try:
        release_month = datetime.strptime(release_date, "%d %b %Y").month
    except ValueError:
        return "Unknown"

    season_preferences = {
        "Summer": ["Action", "Adventure", "Superhero"],
        "Winter": ["Family", "Holiday", "Animation"],
        "Spring": ["Comedy", "Romance", "Action"],
        "Fall": ["Drama", "Biography", "Historical"]
    }

    if release_month in [6, 7, 8]:
        season = "Summer"
    elif release_month in [11, 12]:
        season = "Winter"
    elif release_month in [3, 4, 5]:
        season = "Spring"
    elif release_month in [9, 10]:
        season = "Fall"
    else:
        season = "Unknown"

    for preferred_genre in season_preferences.get(season, []):
        if preferred_genre in genre:
            return f"Highly Suitable for {season}"
    return f"Less Suitable for {season}"

# Function to calculate Diversity in Themes
def calculate_theme_diversity(plot):
    if not plot:
        return "Unknown"

    diversity_keywords = [
        "inclusion", "representation", "diversity", "equality", "equity",
        "justice", "identity", "culture", "community", "global"
    ]
    theme_count = sum(1 for keyword in diversity_keywords if keyword in plot.lower())

    if theme_count > 5:
        return "High Diversity in Themes"
    elif theme_count > 2:
        return "Moderate Diversity in Themes"
    else:
        return "Low Diversity in Themes"

# Function to calculate Director and Cast Influence
def calculate_director_and_cast_influence(director, cast, historical_data):
    director_movies = historical_data[historical_data['Director'] == director]
    director_score = director_movies['IMDb Rating'].mean() if not director_movies.empty else 5.0

    cast_scores = []
    for actor in cast:
        actor_movies = historical_data[historical_data['Cast'].apply(lambda x: actor in x)]
        actor_score = actor_movies['IMDb Rating'].mean() if not actor_movies.empty else 5.0
        cast_scores.append(actor_score)
    cast_score = sum(cast_scores) / len(cast_scores) if cast_scores else 5.0

    return (director_score * 0.6) + (cast_score * 0.4)

# Function to calculate Budget-to-Revenue Ratio
def calculate_budget_revenue_ratio(budget, revenue):
    if budget == 0 or revenue == 0:
        return 1.5
    return budget / revenue

from textblob import TextBlob

# Function to calculate critics sentiment based on reviews or awards
def calculate_critic_sentiment(awards_text):
    # For simplicity, we will use the presence of positive or negative keywords in awards text
    positive_keywords = ["winner", "award", "critically acclaimed", "best"]
    negative_keywords = ["worst", "flop", "bad", "disappointing"]
    
    # Count positive and negative keywords in the awards text
    positive_count = sum(1 for keyword in positive_keywords if keyword in awards_text.lower())
    negative_count = sum(1 for keyword in negative_keywords if keyword in awards_text.lower())
    
    return positive_count - negative_count  # Positive value means more positive sentiment

# Function to calculate audience sentiment based on IMDb rating
def calculate_audience_sentiment(imdb_rating):
    # Audience sentiment could be interpreted from IMDb rating
    if imdb_rating >= 7.5:
        return 1  # Positive sentiment
    elif imdb_rating >= 5.0:
        return 0  # Neutral sentiment
    else:
        return -1  # Negative sentiment

# Function to compare critics vs audience sentiment
def compare_sentiment(awards_text, imdb_rating):
    critic_sentiment = calculate_critic_sentiment(awards_text)
    audience_sentiment = calculate_audience_sentiment(imdb_rating)
    
    # Compare the sentiments
    if critic_sentiment > 0 and audience_sentiment > 0:
        comparison = "Aligned Positive"
    elif critic_sentiment < 0 and audience_sentiment < 0:
        comparison = "Aligned Negative"
    elif critic_sentiment == audience_sentiment:
        comparison = "Aligned Neutral"
    else:
        comparison = "Discrepant Sentiment"
    
    return comparison

# Integrating this new feature in the movie data processing
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            imdb_rating = float(movie_data.get("imdbRating", 0))
            awards_text = movie_data.get("Awards", "")
            
            # Use the new feature to get critics vs audience sentiment
            sentiment_comparison = compare_sentiment(awards_text, imdb_rating)
            
            # Add the sentiment comparison along with other features to the DataFrame
            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Critics vs Audience Sentiment": sentiment_comparison,
                # Include other features...
            })
    return pd.DataFrame(data)

# Example usage
def main():
    api_key = '121c5367'
    movie_titles = ["Inception", "Frozen", "Avengers: Endgame"]
    result_df = process_movie_data(movie_titles, api_key)
    print(result_df)

if __name__ == "__main__":
    main()
