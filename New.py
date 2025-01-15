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

# Function to estimate cultural influence
def estimate_cultural_influence(plot, awards_text):
    cultural_keywords = ["diversity", "inclusion", "trailblazer", "revolutionary", 
                          "groundbreaking", "historic", "global", "impactful"]
    combined_text = f"{plot} {awards_text}".lower()
    score = sum(1 for keyword in cultural_keywords if keyword in combined_text)
    return "High Cultural Influence" if score > 5 else "Moderate Cultural Influence" if score > 2 else "Low Cultural Influence"

# Function to calculate Diversity Index for cast
def calculate_diversity_index(cast_list):
    if not cast_list:
        return 0.0  # No data, no diversity

    # Dummy dataset for demonstration (you can use APIs or databases for real data)
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

    # Diversity Index is the average of unique categories divided by total actors
    diversity_score = (
        len(ethnicities) + len(genders) + len(nationalities)
    ) / (3 * len(cast_list))

    return round(diversity_score * 100, 2)  # Scale to a percentage

# Update process_movie_data to include Diversity Index
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            cast = movie_data.get("Actors", "").split(",") if movie_data.get("Actors") else []
            diversity_index = calculate_diversity_index(cast)
            # Other features...
            data.append({
                "Title": movie_data.get("Title"),
                "Diversity Index": diversity_index,
                # Add other features here
            })
    return pd.DataFrame(data)

# New Feature: Estimate Cinematography Potential
def estimate_cinematography_potential(imdb_rating, awards_text):
    cinematography_keywords = ["cinematography", "visual effects", "visuals"]
    cinematography_awards = any(keyword in awards_text.lower() for keyword in cinematography_keywords)
    score = imdb_rating * 10 + (20 if cinematography_awards else 0)
    return "High Cinematography Potential" if score > 80 else "Moderate Cinematography Potential" if score > 50 else "Low Cinematography Potential"


# Function to calculate climate suitability indicator
def estimate_climate_suitability(release_date, genre):
    try:
        release_month = datetime.strptime(release_date, "%d %b %Y").month
    except ValueError:
        return "Unknown"

    # Define seasonal preferences
    season_preferences = {
        "Summer": ["Action", "Adventure", "Superhero"],
        "Winter": ["Family", "Holiday", "Animation"],
        "Spring": ["Comedy", "Romance", "Action"],
        "Fall": ["Drama", "Biography", "Historical"]
    }

    # Determine the season
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

    # Check if the genre matches the season's preferences
    for preferred_genre in season_preferences.get(season, []):
        if preferred_genre in genre:
            return f"Highly Suitable for {season}"
    return f"Less Suitable for {season}"

# Update process_movie_data to include Climate Suitability
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
            climate_suitability = estimate_climate_suitability(movie_data.get("Released", ""), movie_data.get("Genre", ""))

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
                "Climate Suitability": climate_suitability,
            })
    
    return pd.DataFrame(data)

# Example usage
def main():
    api_key = "your_api_key_here"
    movie_titles = ["Inception", "Frozen", "Avengers: Endgame"]
    result_df = process_movie_data(movie_titles, api_key)
    print(result_df)

if __name__ == "__main__":
    main()
