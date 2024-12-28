import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob

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
        import re
        match = re.search(r"(\d+) win", awards_text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0

# Function to simulate streaming platform availability
def simulate_streaming_availability(title):
    platforms = ["Netflix", "Amazon Prime", "Hulu", "Disney+"]
    import random
    available_platforms = random.sample(platforms, random.randint(1, len(platforms)))
    return available_platforms

# Main function to process movie data
def process_movie_data(titles, api_key):
    data = []
    for title in titles:
        movie_data = fetch_movie_data(title, api_key)
        if movie_data and movie_data.get("Response") == "True":
            genre_sentiment = analyze_genre_sentiment(movie_data.get("Genre", ""))
            holiday_release = is_holiday_release(movie_data.get("Released", ""))
            awards_count = extract_awards_count(movie_data.get("Awards", ""))
            streaming_platforms = simulate_streaming_availability(title)

            data.append({
                "Title": movie_data.get("Title"),
                "Year": movie_data.get("Year"),
                "Genre Sentiment": genre_sentiment,
                "Holiday Release": holiday_release,
                "Awards Count": awards_count,
                "Streaming Platforms": ", ".join(streaming_platforms),
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
