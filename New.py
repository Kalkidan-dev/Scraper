import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import re

# Function to fetch movie data from OMDb API
def fetch_movie_data(title, api_key="your_api_key_here"):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Existing Features
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
    }
    ethnicities, genders, nationalities = set(), set(), set()
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

# New Feature: Comparing Critics vs Audience Sentiment
def calculate_critic_sentiment(awards_text):
    positive_keywords = ["winner", "award", "critically acclaimed", "best"]
    negative_keywords = ["worst", "flop", "bad", "disappointing"]
    positive_count = sum(1 for keyword in positive_keywords if keyword in awards_text.lower())
    negative_count = sum(1 for keyword in negative_keywords if keyword in awards_text.lower())
    return positive_count - negative_count  # Positive value means more positive sentiment

def calculate_audience_sentiment(imdb_rating):
    if imdb_rating >= 7.5:
        return 1  # Positive sentiment
    elif imdb_rating >= 5.0:
        return 0  # Neutral sentiment
    else:
        return -1  # Negative sentiment

def compare_sentiment(awards_text, imdb_rating):
    critic_sentiment = calculate_critic_sentiment(awards_text)
    audience_sentiment = calculate_audience_sentiment(imdb_rating)
    if critic_sentiment > 0 and audience_sentiment > 0:
        return "Aligned Positive"
    elif critic_sentiment < 0 and audience_sentiment < 0:
        return "Aligned Negative"
    elif critic_sentiment == audience_sentiment:
        return "Aligned Neutral"
    else:
        return "Discrepant Sentiment"
# Function to calculate Box Office Analysis
def box_office_analysis(budget, revenue):
    if not budget or not revenue or budget == 0:
        return "Unknown"
    profit_margin = (revenue - budget) / budget
    if profit_margin > 1.0:
        return "High Success"
    elif profit_margin > 0:
        return "Moderate Success"
    else:
        return "Flop"

# Updating the process_movie_data function to include the feature
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            imdb_rating = float(movie_data.get("imdbRating", 0))
            awards_text = movie_data.get("Awards", "")
            budget = float(movie_data.get("Budget", 0))  # Replace with actual budget source
            revenue = float(movie_data.get("BoxOffice", 0))  # Replace with actual revenue source
            
            sentiment_comparison = compare_sentiment(awards_text, imdb_rating)
            box_office_result = box_office_analysis(budget, revenue)
            
            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Critics vs Audience Sentiment": sentiment_comparison,
                "Box Office Analysis": box_office_result,
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
