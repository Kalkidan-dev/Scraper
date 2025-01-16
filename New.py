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

# New Feature: Estimate Marketing Effectiveness
def estimate_marketing_effectiveness(imdb_rating, social_media_buzz, box_office_success):
    score = (imdb_rating * 6 + (10 if social_media_buzz == "High Buzz Potential" else 0) + (10 if box_office_success == "High Box Office Success" else 0))
    return "Highly Effective Marketing" if score > 80 else "Moderately Effective Marketing" if score > 50 else "Ineffective Marketing"

def estimate_cinematography_potential(imdb_rating, awards_text):
    cinematography_keywords = ["cinematography", "visual effects", "visuals"]
    cinematography_awards = any(keyword in awards_text.lower() for keyword in cinematography_keywords)
    score = imdb_rating * 10 + (20 if cinematography_awards else 0)
    return "High Cinematography Potential" if score > 80 else \
           "Moderate Cinematography Potential" if score > 50 else \
           "Low Cinematography Potential"


# Function to calculate climate suitability indicators
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
            # Existing processing (genre sentiment, awards count, etc.)
            imdb_rating = float(movie_data.get("imdbRating", 0))
            genre_sentiment = analyze_genre_sentiment(movie_data.get("Genre", ""))
            plot_sentiment = TextBlob(movie_data.get("Plot", "")).sentiment.polarity if movie_data.get("Plot") else 0
            awards_count = extract_awards_count(movie_data.get("Awards", ""))
            social_media_buzz = estimate_social_media_buzz(int(movie_data.get("Year", 0)), imdb_rating, genre_sentiment)
            box_office_success = estimate_box_office_success(imdb_rating, awards_count, genre_sentiment)
            rewatch_value = estimate_rewatch_value(imdb_rating, plot_sentiment, awards_count)
            # New features
            audience_appeal = estimate_audience_appeal(imdb_rating, genre_sentiment, awards_count)
            long_term_streaming_popularity = estimate_long_term_streaming_popularity(imdb_rating, genre_sentiment, social_media_buzz)
            international_appeal = estimate_international_appeal(imdb_rating, calculate_diversity_index(movie_data.get("Actors", "").split(",")), movie_data.get("Genre", ""))
            critical_reception = estimate_critical_reception(imdb_rating, plot_sentiment)
            viewer_retention = estimate_viewer_retention(imdb_rating, rewatch_value, social_media_buzz)
            storyline_impact = estimate_storyline_impact(imdb_rating, plot_sentiment, genre_sentiment)
            marketing_effectiveness = estimate_marketing_effectiveness(imdb_rating, social_media_buzz, box_office_success)
            cinematography_potential = estimate_cinematography_potential(imdb_rating, movie_data.get("Awards", ""))

            data.append({
                "Title": movie_data.get("Title"),
                "IMDb Rating": imdb_rating,
                "Genre Sentiment": genre_sentiment,
                "Awards Count": awards_count,
                "Social Media Buzz": social_media_buzz,
                "Box Office Success": box_office_success,
                "Rewatch Value": rewatch_value,
                "Audience Appeal": audience_appeal,
                "Long-Term Streaming Popularity": long_term_streaming_popularity,
                "International Appeal": international_appeal,
                "Critical Reception": critical_reception,
                "Viewer Retention": viewer_retention,
                "Storyline Impact": storyline_impact,
                "Marketing Effectiveness": marketing_effectiveness,
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
