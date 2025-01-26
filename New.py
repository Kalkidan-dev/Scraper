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

import random

# Function to simulate social media buzz analysis
def analyze_social_media_buzz(title):
    # Dummy implementation: Randomly generate social media metrics
    mentions = random.randint(1000, 100000)  # Number of mentions
    hashtags = random.randint(100, 10000)    # Number of unique hashtags
    sentiment_score = random.uniform(-1, 1)  # Sentiment score (-1 to 1)

    # Calculate buzz score (weighted formula)
    buzz_score = (0.4 * (mentions / 100000)) + (0.3 * (hashtags / 10000)) + (0.3 * (sentiment_score + 1) / 2)
    return round(buzz_score * 100, 2)  # Scale to percentage

# Function to calculate streaming platform compatibility
def calculate_streaming_compatibility(runtime, genres):
    # Define preferred genres for streaming platforms
    streaming_preferred_genres = [
        "Drama", "Comedy", "Thriller", "Documentary", "Romance", "Sci-Fi"
    ]

    # Score based on genres
    genre_score = sum(1 for genre in genres.split(",") if genre.strip() in streaming_preferred_genres)
    genre_score = genre_score / len(streaming_preferred_genres)

    # Score based on runtime (optimal streaming runtime: 90-120 minutes)
    runtime_minutes = int(re.search(r"(\d+)", runtime).group(1)) if runtime else 0
    if 90 <= runtime_minutes <= 120:
        runtime_score = 1.0
    elif 60 <= runtime_minutes < 90 or 120 < runtime_minutes <= 150:
        runtime_score = 0.7
    else:
        runtime_score = 0.4

    # Combine genre and runtime scores
    compatibility_score = (0.6 * genre_score) + (0.4 * runtime_score)
    return round(compatibility_score * 100, 2)

# Function to check streaming platform availability
def check_streaming_availability(movie_title):
    # Simulated dataset for demonstration
    streaming_data = {
        "Inception": ["Netflix", "Amazon Prime"],
        "Frozen": ["Disney+"],
        "Avengers: Endgame": ["Disney+", "Amazon Prime"],
        # Add more movies as needed
    }
    
    return streaming_data.get(movie_title, ["Not Available"])

# Function to estimate target audience age group
def estimate_audience_age_group(genre, plot, mpaa_rating):
    # Define genre and plot-based age group preferences
    age_group_preferences = {
        "Kids": ["Animation", "Family", "Adventure"],
        "Teens": ["Action", "Comedy", "Fantasy"],
        "Adults": ["Drama", "Thriller", "Horror"],
        "All Ages": ["Biography", "Historical", "Documentary"]
    }

    # Default MPAA-based age group
    if mpaa_rating in ["G"]:
        return "Kids"
    elif mpaa_rating in ["PG"]:
        return "Kids to Teens"
    elif mpaa_rating in ["PG-13"]:
        return "Teens to Adults"
    elif mpaa_rating in ["R", "NC-17"]:
        return "Adults"
    else:
        mpaa_based_group = "All Ages"

    # Determine age group based on genre and plot
    for age_group, genres in age_group_preferences.items():
        for keyword in genres:
            if keyword in genre or keyword in plot:
                return age_group

    # Fall back to MPAA-based group if no match
    return mpaa_based_group

# Integrate the feature into the process_movie_data function
# Function to calculate language influence score
def calculate_language_influence(languages):
    if not languages:
        return 0.0

    # Assign influence scores to common languages (example values, can be adjusted)
    language_influence = {
        "English": 1.0,
        "Mandarin": 0.9,
        "Spanish": 0.8,
        "Hindi": 0.7,
        "French": 0.6,
        "Arabic": 0.6,
        "Portuguese": 0.5,
        "Russian": 0.5,
        "Japanese": 0.5,
        "German": 0.4,
        # Add more languages if needed
    }

    score = 0.0
    language_list = [lang.strip() for lang in languages.split(",")]

    for lang in language_list:
        score += language_influence.get(lang, 0.2)  # Default influence score for less common languages

    # Normalize the score based on the number of languages
    return round(score / len(language_list), 2)

# Function to calculate age group suitability
def calculate_age_group_suitability(genre, content_rating):
    if not genre or not content_rating:
        return "Unknown"

    # Define age group preferences based on genres
    age_group_preferences = {
        "Children": ["Animation", "Family", "Fantasy", "Adventure"],
        "Teenagers": ["Action", "Romance", "Comedy", "Superhero"],
        "Adults": ["Drama", "Thriller", "Horror", "Crime"],
        "Seniors": ["Biography", "Historical", "Documentary"]
    }

    # Match content rating to broad age groups
    content_rating_mapping = {
        "G": "Children",
        "PG": "Children/Teenagers",
        "PG-13": "Teenagers/Adults",
        "R": "Adults",
        "NC-17": "Adults",
        # Add more ratings if applicable
    }

    # Determine the primary audience from content rating
    primary_audience = content_rating_mapping.get(content_rating, "Unknown")

    # Check genre alignment with each age group
    suitability = []
    for age_group, preferred_genres in age_group_preferences.items():
        for preferred_genre in preferred_genres:
            if preferred_genre in genre:
                suitability.append(age_group)

    if not suitability:
        return f"Suitable for {primary_audience}" if primary_audience != "Unknown" else "Unknown"
    return f"Suitable for {', '.join(set(suitability))}"

# Function to calculate runtime suitability for different age groups
def calculate_runtime_suitability(runtime, age_group):
    if not runtime or not age_group:
        return "Unknown"

    runtime_minutes = int(runtime.replace(" min", "")) if "min" in runtime else 0

    # Define typical runtime preferences by age group
    runtime_preferences = {
        "Children": (30, 90),  # Suitable for 30-90 mins
        "Teenagers": (90, 150),  # Suitable for 90-150 mins
        "Adults": (90, 180),  # Suitable for 90-180 mins
        "Seniors": (60, 150),  # Suitable for 60-150 mins
    }

    for group, (min_time, max_time) in runtime_preferences.items():
        if group in age_group and min_time <= runtime_minutes <= max_time:
            return "Suitable"
    return "Not Suitable"

# Function to predict if a movie is a blockbuster
def predict_blockbuster(imdb_rating, awards, genre, box_office):
    # Define thresholds for a blockbuster
    min_imdb_rating = 7.5
    min_awards = 5
    high_box_office = 100_000_000  # Example: $100M
    
    # Genre-based popularity (example scores)
    genre_popularity = {
        "Action": 1.2,
        "Adventure": 1.1,
        "Comedy": 1.0,
        "Drama": 0.9,
        "Horror": 0.8,
        "Romance": 0.8,
        "Animation": 1.3,
        # Add more genres as needed
    }
    
    # Calculate genre influence
    genre_score = sum(genre_popularity.get(g.strip(), 1.0) for g in genre.split(","))
    
    # Normalize box office revenue
    box_office = int(box_office.replace("$", "").replace(",", "").strip()) if box_office else 0
    
    # Predict blockbuster status
    if (
        float(imdb_rating) >= min_imdb_rating
        and int(awards) >= min_awards
        and box_office >= high_box_office * genre_score
    ):
        return "Yes"
    return "No"

from datetime import datetime

# Function to categorize release season
def get_release_season(release_date):
    if not release_date:
        return "Unknown"
    
    # Parse the release date (assuming format: "DD MMM YYYY")
    try:
        date_object = datetime.strptime(release_date, "%d %b %Y")
    except ValueError:
        return "Unknown"
    
    month = date_object.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    return "Unknown"

from textblob import TextBlob

# Function to perform sentiment analysis on a movie plot
def analyze_plot_sentiment(plot):
    if not plot:
        return "Unknown"
    
    # Analyze sentiment using TextBlob
    sentiment = TextBlob(plot).sentiment
    if sentiment.polarity > 0.2:
        return "Positive"
    elif sentiment.polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Function to calculate director's popularity score
def calculate_director_popularity(director, imdb_rating, awards, box_office):
    if not director:
        return 0

    # Define thresholds for successful movies
    min_imdb_rating = 7.0
    min_awards = 3
    high_box_office = 50_000_000  # Example: $50M
    
    # Calculate box office revenue
    box_office = int(box_office.replace("$", "").replace(",", "").strip()) if box_office else 0
    
    # Define a basic "popularity score" for each movie based on IMDb rating, awards, and box office
    popularity_score = 0
    if float(imdb_rating) >= min_imdb_rating:
        popularity_score += 2
    if int(awards) >= min_awards:
        popularity_score += 2
    if box_office >= high_box_office:
        popularity_score += 1
    
    return popularity_score

# Function to determine if a movie is part of a popular franchise
def is_movie_in_franchise(title):
    # Define popular franchises (you can add more as needed)
    franchises = ["Marvel", "Star Wars", "Harry Potter", "Jurassic Park", "Fast & Furious", "Mission: Impossible", "Pirates of the Caribbean", "Transformers", "The Lord of the Rings"]

    # Check if any franchise name is part of the movie title (case-insensitive)
    for franchise in franchises:
        if franchise.lower() in title.lower():
            return True
    return False

# Function to calculate budget to box office ratio
def calculate_budget_to_box_office_ratio(budget, box_office):
    if not budget or not box_office:
        return "Unknown"

    # Remove non-numeric characters (currency symbols, commas, etc.)
    budget = int(budget.replace("$", "").replace(",", "").strip()) if budget else 0
    box_office = int(box_office.replace("$", "").replace(",", "").strip()) if box_office else 0

    # Avoid division by zero
    if box_office == 0:
        return "No Box Office Data"
    
    # Calculate the ratio
    ratio = box_office / budget
    return round(ratio, 2)

# Function to check if a movie is a sequel
def is_sequel(title):
    # Look for patterns in the title that indicate a sequel
    sequel_patterns = [
        r"\b2\b",            # Matches "2" (e.g., "Frozen 2")
        r"\bII\b",           # Matches Roman numeral II
        r"\bIII\b",          # Matches Roman numeral III
        r"\bIV\b",           # Matches Roman numeral IV
        r":",                # Matches colon, often used for subtitles
        r"\bPart \d+\b",     # Matches "Part 2", "Part 3", etc.
        r"\bChapter \d+\b"   # Matches "Chapter 2", "Chapter 3", etc.
    ]
    
    # Check if any pattern matches the title
    for pattern in sequel_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return True
    return False

# Function to determine the season of a movie's release
def determine_season(release_date):
    if not release_date:
        return "Unknown"
    
    # Parse the release date
    try:
        release_month = pd.to_datetime(release_date).month
    except ValueError:
        return "Unknown"
    
    # Map months to seasons
    if release_month in [12, 1, 2]:
        return "Winter"
    elif release_month in [3, 4, 5]:
        return "Spring"
    elif release_month in [6, 7, 8]:
        return "Summer"
    elif release_month in [9, 10, 11]:
        return "Fall"
    return "Unknown"

# Function to calculate critic vs. audience score gap
def calculate_score_gap(critic_score, audience_score):
    if not critic_score or not audience_score:
        return "Unknown"
    
    try:
        critic_score = int(critic_score.strip('%')) if "%" in critic_score else int(critic_score)
        audience_score = float(audience_score) * 10 if 0 <= float(audience_score) <= 10 else int(audience_score)
    except ValueError:
        return "Unknown"

    # Calculate the absolute difference
    return abs(critic_score - audience_score)

# Function to classify box office success
def classify_box_office_success(box_office):
    if not box_office:
        return "Unknown"
    
    # Remove special characters like $ and commas
    try:
        box_office = float(box_office.replace("$", "").replace(",", ""))
    except ValueError:
        return "Unknown"
    
    # Define thresholds for classification
    if box_office >= 1_000_000_000:  # 1 billion or more
        return "Blockbuster"
    elif box_office >= 500_000_000:  # Between 500 million and 1 billion
        return "Hit"
    elif box_office >= 100_000_000:  # Between 100 million and 500 million
        return "Average"
    else:
        return "Flop"

# Simulated dataset for director popularity (can be replaced with real data)
director_popularity_data = {
    "Steven Spielberg": 95,
    "Christopher Nolan": 90,
    "James Cameron": 92,
    "Quentin Tarantino": 88,
    "Martin Scorsese": 85,
    "Greta Gerwig": 80,
    # Add more directors as needed
}

# Function to get director popularity score
def get_director_popularity_score(director):
    if not director:
        return 50  # Default score for unknown directors
    return director_popularity_data.get(director, 60)  # Return default for less-known directors

import re

# Function to check if a movie is a sequel
def is_sequel(title):
    if not title:
        return "Unknown"
    
    # Check for common patterns indicating a sequel
    sequel_patterns = [
        r"\b[2-9]$",             # Titles ending with a number (e.g., "Toy Story 2")
        r"[2-9]:",               # Titles with a number and colon (e.g., "Rocky 4: The Final Round")
        r"Part\s[2-9]",          # Titles with "Part" followed by a number (e.g., "Harry Potter Part 2")
        r"Chapter\s[2-9]",       # Titles with "Chapter" followed by a number (e.g., "It Chapter 2")
        r"Volume\s[2-9]",        # Titles with "Volume" followed by a number (e.g., "Kill Bill Volume 2")
        r"Revenge|Return|Saga",  # Keywords indicating sequels or franchises
        r"Finale",               # Keywords indicating franchise conclusions
    ]
    
    for pattern in sequel_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return "Sequel"
    
    return "Standalone"

# Simulated dataset for production budgets (can be replaced with real data or API integration)
movie_budget_data = {
    "Inception": 160_000_000,
    "Frozen": 150_000_000,
    "Avengers: Endgame": 356_000_000,
    "The Dark Knight": 185_000_000,
    "The Godfather": 6_000_000,
    # Add more movies as needed
}

# Function to get a movie's production budget
def get_production_budget(title):
    if not title:
        return "Unknown"
    return movie_budget_data.get(title, "Estimate Unavailable")

# Function to estimate budget impact category
def estimate_budget_impact(budget):
    if budget == "Unknown" or budget == "Estimate Unavailable":
        return "Insufficient Data"
    if budget <= 10_000_000:
        return "Low Budget"
    elif budget <= 50_000_000:
        return "Moderate Budget"
    elif budget <= 150_000_000:
        return "High Budget"
    else:
        return "Blockbuster Budget"

from textblob import TextBlob  # Ensure you have TextBlob installed: pip install textblob

# Simulated dataset of critic reviews
critic_reviews_data = {
    "Inception": [
        "A visually stunning masterpiece with mind-bending concepts.",
        "Nolan's brilliance shines through, though it can be hard to follow at times.",
        "A must-watch, but it requires attention to fully grasp the plot."
    ],
    "Frozen": [
        "A heartwarming story for kids and adults alike.",
        "The songs are memorable, but the story feels predictable at times.",
        "Disney does it again with another magical movie."
    ],
    "Avengers: Endgame": [
        "A satisfying conclusion to a decade-long journey.",
        "Epic battles, emotional moments, and great character arcs.",
        "A few pacing issues, but overall a cinematic triumph."
    ],
    # Add more reviews for other movies as needed
}

# Function to analyze review sentiment
def analyze_review_sentiment(reviews):
    if not reviews:
        return {"Sentiment": "No Reviews", "Score": 0.0}
    
    sentiment_score = 0.0
    for review in reviews:
        analysis = TextBlob(review)
        sentiment_score += analysis.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
    
    average_score = sentiment_score / len(reviews)
    sentiment = (
        "Positive" if average_score > 0.1 else
        "Negative" if average_score < -0.1 else
        "Neutral"
    )
    
    return {"Sentiment": sentiment, "Score": round(average_score, 2)}

# Function to analyze language popularity
def analyze_language_popularity(languages):
    if not languages:
        return "Unknown"

    # Example dataset for language popularity (audience size approximation)
    language_popularity = {
        "English": 1000,  # High audience size
        "Spanish": 500,   # Moderate audience size
        "Mandarin": 800,
        "Hindi": 700,
        "French": 400,
        "German": 350,
        "Japanese": 300,
        "Korean": 200,
        # Add more languages as necessary
    }

    total_popularity_score = 0
    languages_found = 0

    for lang in languages.split(","):
        lang = lang.strip()
        if lang in language_popularity:
            total_popularity_score += language_popularity[lang]
            languages_found += 1

    if languages_found == 0:
        return "Low Popularity (Rare Language)"
    avg_score = total_popularity_score / languages_found

    if avg_score > 750:
        return "High Popularity"
    elif avg_score > 400:
        return "Moderate Popularity"
    else:
        return "Low Popularity"

# Function to analyze runtime suitability
def analyze_runtime_suitability(runtime, genres):
    try:
        runtime_minutes = int(runtime.split(" ")[0])  # Extract numeric runtime in minutes
    except (ValueError, AttributeError):
        return "Unknown"

    # Typical runtime ranges for genres (in minutes)
    genre_runtime_preferences = {
        "Action": (90, 150),
        "Drama": (100, 180),
        "Comedy": (80, 120),
        "Horror": (70, 110),
        "Romance": (80, 130),
        "Animation": (60, 120),
        "Documentary": (50, 120),
    }

    suitability_scores = []
    for genre in genres.split(","):
        genre = genre.strip()
        if genre in genre_runtime_preferences:
            min_runtime, max_runtime = genre_runtime_preferences[genre]
            if min_runtime <= runtime_minutes <= max_runtime:
                suitability_scores.append(1)  # Suitable
            else:
                suitability_scores.append(0)  # Not suitable

    if not suitability_scores:
        return "Unknown"

    suitability_ratio = sum(suitability_scores) / len(suitability_scores)
    if suitability_ratio == 1:
        return "Highly Suitable"
    elif suitability_ratio >= 0.5:
        return "Moderately Suitable"
    else:
        return "Less Suitable"

# Function to evaluate language popularity
def evaluate_language_popularity(language):
    # Popularity score based on global audience preferences (1-10 scale)
    language_popularity = {
        "English": 10,
        "Spanish": 8,
        "French": 7,
        "Mandarin": 9,
        "Hindi": 8,
        "Japanese": 7,
        "German": 6,
        "Korean": 8,
        "Italian": 6,
        "Russian": 5,
        # Add more languages as needed
    }

    return language_popularity.get(language, 4)  # Default score for unknown languages

# Function to evaluate runtime suitability
def evaluate_runtime_suitability(runtime, genres):
    if not runtime or not genres:
        return "Unknown"

    try:
        runtime_minutes = int(runtime.split()[0])  # Extract runtime in minutes
    except ValueError:
        return "Unknown"

    # Define ideal runtime ranges for various genres (in minutes)
    genre_runtime_ranges = {
        "Action": (90, 150),
        "Comedy": (75, 120),
        "Drama": (90, 180),
        "Horror": (70, 110),
        "Animation": (80, 120),
        "Documentary": (60, 120),
        "Romance": (75, 130),
        # Add more genres and their ranges as needed
    }

    # Check runtime suitability for all genres in the movie
    suitability = []
    for genre in genres.split(","):
        genre = genre.strip()
        if genre in genre_runtime_ranges:
            min_runtime, max_runtime = genre_runtime_ranges[genre]
            if min_runtime <= runtime_minutes <= max_runtime:
                suitability.append(f"Suitable for {genre}")
            else:
                suitability.append(f"Long/Short for {genre}")

    return ", ".join(suitability) if suitability else "Unknown"

# Function to analyze franchise continuity
def analyze_franchise_continuity(title, release_year, franchise_data):
    if not franchise_data:
        return "Unknown"

    franchise_info = franchise_data.get(title.lower(), {})
    if not franchise_info:
        return "Not Part of Franchise"

    # Check continuity by comparing release years with previous titles
    previous_titles = franchise_info.get("previous_titles", [])
    previous_years = franchise_info.get("release_years", [])

    if previous_titles and previous_years:
        latest_previous_year = max(previous_years)
        if int(release_year) - latest_previous_year <= 3:
            return "Strong Continuity"
        else:
            return "Weak Continuity"
    return "Part of Franchise, No Continuity Data"

# Sample franchise data for demonstration
franchise_data = {
    "avengers: endgame": {
        "previous_titles": ["Avengers: Infinity War", "Avengers: Age of Ultron"],
        "release_years": [2018, 2015],
    },
    "toy story 4": {
        "previous_titles": ["Toy Story 3", "Toy Story 2"],
        "release_years": [2010, 1999],
    },
}

# Integrate the feature into the process_movie_data function
def process_movie_data_with_franchise_analysis(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            release_year = movie_data.get("Year", "0").split("–")[0]
            
            # Use the new feature to analyze franchise continuity
            franchise_continuity = analyze_franchise_continuity(
                title, release_year, franchise_data
            )

            data.append({
                "Title": movie_data.get("Title"),
                "Year": release_year,
                "Franchise Continuity": franchise_continuity,
                # Include other features...
            })
    return pd.DataFrame(data)

# Example usage
def main():
    api_key = '121c5367'
    movie_titles = ["Avengers: Endgame", "Toy Story 4", "Inception"]
    result_df = process_movie_data_with_franchise_analysis(movie_titles, api_key)
    print(result_df)

if __name__ == "__main__":
    main()
