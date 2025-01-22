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

# Function to calculate Franchise Potential Score
def calculate_franchise_potential(genre, revenue, imdb_rating):
    franchise_genres = ["Action", "Adventure", "Sci-Fi", "Superhero", "Fantasy", "Animation"]
    genre_factor = any(g in genre for g in franchise_genres)
    
    # Normalize revenue for scoring (example: divide by 1 billion to scale it)
    revenue_score = revenue / 1_000_000_000 if revenue else 0
    
    # Adjust IMDb rating for scoring
    imdb_score = imdb_rating / 10 if imdb_rating else 0
    
    # Franchise potential formula
    franchise_score = (0.5 * genre_factor) + (0.3 * revenue_score) + (0.2 * imdb_score)
    return round(franchise_score * 100, 2)  # Scale to percentage

# Function to calculate Audience Demographics Appeal
def calculate_audience_demographics_appeal(genres):
    demographic_preferences = {
        "Family": {"Kids": 0.7, "Teens": 0.5, "Adults": 0.3},
        "Action": {"Kids": 0.4, "Teens": 0.8, "Adults": 0.7},
        "Romance": {"Teens": 0.6, "Adults": 0.8},
        "Comedy": {"Kids": 0.6, "Teens": 0.7, "Adults": 0.6},
        "Drama": {"Adults": 0.9},
        "Animation": {"Kids": 0.9, "Teens": 0.6}
    }
    appeal_scores = {"Kids": 0, "Teens": 0, "Adults": 0}
    for genre in genres.split(","):
        genre = genre.strip()
        if genre in demographic_preferences:
            for group, score in demographic_preferences[genre].items():
                appeal_scores[group] += score
    # Normalize scores
    total_genres = len(genres.split(","))
    for group in appeal_scores:
        appeal_scores[group] = round(appeal_scores[group] / total_genres, 2)
    return appeal_scores

# Function to calculate Genre Diversity Score
def calculate_genre_diversity(genres):
    if not genres:
        return 0
    genre_list = genres.split(",")
    unique_genres = set(genre.strip() for genre in genre_list)
    total_genres = len(genre_list)
    diversity_score = len(unique_genres) / total_genres if total_genres > 0 else 0
    return round(diversity_score * 100, 2)

# Function to calculate Star Power Index
def calculate_star_power_index(cast_list):
    if not cast_list:
        return 0.0
    
    # Dummy dataset for actor popularity (for demonstration)
    actor_popularity = {
        "Leonardo DiCaprio": 95,
        "Morgan Freeman": 90,
        "Marion Cotillard": 85,
        "Robert Downey Jr.": 98,
        "Scarlett Johansson": 92,
        # Add more actors and their popularity scores
    }
    
    total_score = 0
    actor_count = 0
    
    for actor in cast_list:
        actor_score = actor_popularity.get(actor.strip(), 50)  # Default score for unknown actors
        total_score += actor_score
        actor_count += 1
    
    # Average star power score
    return round(total_score / actor_count, 2) if actor_count > 0 else 0.0

# Function to calculate Soundtrack Popularity Score
def calculate_soundtrack_popularity(soundtrack_artists):
    if not soundtrack_artists:
        return 0.0
    
    # Dummy dataset for artist popularity (for demonstration)
    artist_popularity = {
        "Hans Zimmer": 95,
        "John Williams": 92,
        "Beyoncé": 98,
        "Adele": 97,
        "Taylor Swift": 96,
        # Add more artists and their popularity scores
    }
    
    total_score = 0
    artist_count = 0
    
    for artist in soundtrack_artists.split(","):
        artist_score = artist_popularity.get(artist.strip(), 50)  # Default score for unknown artists
        total_score += artist_score
        artist_count += 1
    
    # Average soundtrack popularity score
    return round(total_score / artist_count, 2) if artist_count > 0 else 0.0

# Function to calculate Sequel Potential Score
def calculate_sequel_potential(genre, revenue, imdb_rating, is_part_of_franchise):
    sequel_genres = ["Action", "Adventure", "Sci-Fi", "Fantasy", "Animation", "Superhero"]
    genre_factor = any(g in genre for g in sequel_genres)
    
    # Normalize revenue for scoring (e.g., divide by 1 billion to scale it)
    revenue_score = revenue / 1_000_000_000 if revenue else 0
    
    # Adjust IMDb rating for scoring
    imdb_score = imdb_rating / 10 if imdb_rating else 0
    
    # Consider if the movie is already part of a franchise
    franchise_factor = 1 if is_part_of_franchise else 0.5
    
    # Sequel potential formula
    sequel_score = (0.4 * genre_factor) + (0.3 * revenue_score) + (0.2 * imdb_score) + (0.1 * franchise_factor)
    return round(sequel_score * 100, 2)  # Scale to percentage

# Function to calculate Social Media Buzz Score
def calculate_social_media_buzz(title):
    # Dummy dataset for demonstration
    social_media_data = {
        "Inception": {"mentions": 150_000, "engagements": 2_500_000},
        "Frozen": {"mentions": 200_000, "engagements": 3_000_000},
        "Avengers: Endgame": {"mentions": 500_000, "engagements": 10_000_000},
    }
    buzz_data = social_media_data.get(title, {"mentions": 0, "engagements": 0})
    
    # Calculate Buzz Score
    mentions_score = buzz_data["mentions"] / 1_000_000  # Normalize mentions
    engagements_score = buzz_data["engagements"] / 10_000_000  # Normalize engagements
    buzz_score = (0.6 * mentions_score) + (0.4 * engagements_score)
    
    return round(buzz_score * 100, 2)  # Scale to percentage

# Function to calculate Sequel Potential Score
def calculate_sequel_potential(genres, revenue, imdb_rating, is_part_of_franchise):
    sequel_genres = ["Action", "Adventure", "Sci-Fi", "Fantasy", "Animation"]
    genre_factor = any(genre.strip() in sequel_genres for genre in genres.split(","))
    
    # Normalize revenue and IMDb rating for scoring
    revenue_score = revenue / 1_000_000_000 if revenue else 0  # Scale revenue
    imdb_score = imdb_rating / 10 if imdb_rating else 0  # Scale IMDb rating

    # Sequel potential formula
    sequel_score = (
        (0.5 * genre_factor) +
        (0.3 * revenue_score) +
        (0.1 * imdb_score) +
        (0.1 if is_part_of_franchise else 0)
    )
    return round(sequel_score * 100, 2)  # Scale to percentage

# Function to calculate Social Media Buzz Score
def calculate_social_media_buzz(movie_title):
    # Dummy dataset for social media analysis
    social_media_data = {
        "Inception": {"mentions": 120000, "hashtags": 45000, "likes": 800000},
        "Frozen": {"mentions": 200000, "hashtags": 60000, "likes": 1000000},
        "Avengers: Endgame": {"mentions": 500000, "hashtags": 150000, "likes": 3000000},
    }

    movie_stats = social_media_data.get(movie_title, None)
    if not movie_stats:
        return "Unknown"

    # Normalize values for scoring
    mention_score = movie_stats["mentions"] / 1_000_000
    hashtag_score = movie_stats["hashtags"] / 100_000
    like_score = movie_stats["likes"] / 10_000_000

    # Social media buzz formula
    buzz_score = (0.4 * mention_score) + (0.3 * hashtag_score) + (0.3 * like_score)
    return round(buzz_score * 100, 2)  # Scale to percentage

# Function to analyze sequels and prequels
def analyze_sequels_and_prequels(movie_title):
    # Dummy dataset for sequels and prequels analysis
    franchise_data = {
        "Inception": {"is_sequel": False, "related_movies": []},
        "Frozen": {"is_sequel": True, "related_movies": [{"title": "Frozen II", "rating": 7.0}]},
        "Avengers: Endgame": {
            "is_sequel": True,
            "related_movies": [
                {"title": "Avengers: Infinity War", "rating": 8.4},
                {"title": "Avengers", "rating": 8.0},
            ],
        },
    }

    movie_info = franchise_data.get(movie_title, None)
    if not movie_info:
        return "Unknown"

    is_sequel = movie_info["is_sequel"]
    related_movies = movie_info["related_movies"]

    # Calculate the average IMDb rating of related movies
    if related_movies:
        avg_related_rating = sum(movie["rating"] for movie in related_movies) / len(related_movies)
    else:
        avg_related_rating = 0

    # Assign a score: Higher for sequels with highly rated related movies
    score = 50 + (avg_related_rating * 5) if is_sequel else 20
    return round(score, 2)

# Function to calculate Social Media Buzz Score
def calculate_social_media_buzz(movie_title):
    # Dummy dataset for demonstration purposes
    social_media_data = {
        "Inception": {"mentions": 150000, "trending_score": 85},
        "Frozen": {"mentions": 200000, "trending_score": 90},
        "Avengers: Endgame": {"mentions": 500000, "trending_score": 95},
    }

    movie_data = social_media_data.get(movie_title, None)
    if not movie_data:
        return "Unknown"

    # Calculate buzz score as a weighted average of mentions and trending score
    mentions_score = movie_data["mentions"] / 10000  # Normalize mentions
    trending_score = movie_data["trending_score"] * 0.5  # Weight trending score
    buzz_score = round(mentions_score + trending_score, 2)

    return min(buzz_score, 100)  # Cap score at 100

# Function to calculate Soundtrack Popularity Score
def calculate_soundtrack_popularity(movie_title):
    # Dummy dataset for demonstration purposes
    soundtrack_data = {
        "Inception": {"album_sales": 200000, "chart_rank": 5},
        "Frozen": {"album_sales": 1000000, "chart_rank": 1},
        "Avengers: Endgame": {"album_sales": 500000, "chart_rank": 3},
    }

    movie_data = soundtrack_data.get(movie_title, None)
    if not movie_data:
        return "Unknown"

    # Calculate popularity score based on album sales and chart rank
    sales_score = movie_data["album_sales"] / 100000  # Normalize album sales
    rank_score = (10 - movie_data["chart_rank"]) * 10  # Higher rank is better
    soundtrack_score = round((sales_score * 0.6) + (rank_score * 0.4), 2)

    return min(soundtrack_score, 100)  # Cap score at 100

# Function to calculate Sequel Potential Score
def calculate_sequel_potential(genre, revenue, imdb_rating):
    # Genres commonly associated with sequels
    sequel_genres = ["Action", "Adventure", "Sci-Fi", "Fantasy", "Superhero", "Animation"]
    genre_factor = any(g in genre for g in sequel_genres)

    # Normalize revenue for scoring (example: divide by 1 billion to scale it)
    revenue_score = revenue / 1_000_000_000 if revenue else 0

    # Adjust IMDb rating for scoring
    imdb_score = imdb_rating / 10 if imdb_rating else 0

    # Sequel potential formula
    sequel_score = (0.4 * genre_factor) + (0.4 * revenue_score) + (0.2 * imdb_score)
    return round(sequel_score * 100, 2)  # Scale to percentage

# Update process_movie_data to include Sequel Potential Score
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
            box_office = movie_data.get("BoxOffice", "")
            revenue = int(re.sub(r"[^\d]", "", box_office)) if box_office else 0

            sentiment_comparison = compare_sentiment(awards_text, imdb_rating)
            viewer_engagement = calculate_viewer_engagement(runtime, genres)
            international_appeal = calculate_international_appeal(languages, countries, cast)
            franchise_potential = calculate_franchise_potential(genres, revenue, imdb_rating)
            sequel_potential = calculate_sequel_potential(genres, revenue, imdb_rating)

            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Critics vs Audience Sentiment": sentiment_comparison,
                "Viewer Engagement": viewer_engagement,
                "International Appeal Score": international_appeal,
                "Franchise Potential Score": franchise_potential,
                "Sequel Potential Score": sequel_potential,
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
