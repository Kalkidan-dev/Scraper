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

# Function to calculate Viewer Engagement
def calculate_viewer_engagement(runtime, genres):
    if not runtime or not genres:
        return "Unknown"
    
    # Categorize runtime
    try:
        runtime = int(runtime.replace(" min", ""))
        if runtime < 90:
            runtime_category = "Short"
        elif 90 <= runtime <= 150:
            runtime_category = "Moderate"
        else:
            runtime_category = "Long"
    except ValueError:
        return "Unknown"
    
    # Assign weights to genres
    high_engagement_genres = ["Drama", "Fantasy", "Mystery", "Sci-Fi"]
    genre_weight = sum(1 for genre in genres.split(",") if genre.strip() in high_engagement_genres)
    
    # Combine metrics
    if runtime_category == "Long" and genre_weight >= 2:
        return "High Engagement"
    elif runtime_category == "Moderate" and genre_weight >= 1:
        return "Moderate Engagement"
    else:
        return "Low Engagement"

# Function to calculate International Appeal Score
def calculate_international_appeal(languages, countries, cast_list):
    if not languages or not countries:
        return "Unknown"
    
    try:
        # Count number of languages
        language_count = len(languages.split(","))
        
        # Count number of production countries
        country_count = len(countries.split(","))
        
        # Dummy dataset for cast details (for demonstration)
        cast_details = {
            "Leonardo DiCaprio": {"nationality": "USA"},
            "Morgan Freeman": {"nationality": "USA"},
            "Marion Cotillard": {"nationality": "France"},
            # Add more actors for real calculations
        }
        
        # Count unique nationalities in the cast
        cast_nationalities = set()
        for actor in cast_list:
            details = cast_details.get(actor.strip())
            if details:
                cast_nationalities.add(details["nationality"])
        cast_nationality_count = len(cast_nationalities)
        
        # Calculate International Appeal Score
        appeal_score = (language_count * 0.4) + (country_count * 0.4) + (cast_nationality_count * 0.2)
        return round(appeal_score, 2)
    
    except Exception as e:
        return "Unknown"

# Updating the process_movie_data function to include the International Appeal Score feature
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            imdb_rating = float(movie_data.get("imdbRating", 0))
            awards_text = movie_data.get("Awards", "")
            runtime = movie_data.get("Runtime", "")
            genres = movie_data.get("Genre", "")
            languages = movie_data.get("Language", "")
            countries = movie_data.get("Country", "")
            cast = movie_data.get("Actors", "").split(",")
            
            sentiment_comparison = compare_sentiment(awards_text, imdb_rating)
            viewer_engagement = calculate_viewer_engagement(runtime, genres)
            international_appeal = calculate_international_appeal(languages, countries, cast)
            
            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Critics vs Audience Sentiment": sentiment_comparison,
                "Viewer Engagement": viewer_engagement,
                "International Appeal Score": international_appeal,
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
