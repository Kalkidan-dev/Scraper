import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from datetime import datetime
import re

# Load OMDb API key from environment variable
api_key = os.getenv("OMDB_API_KEY")

if not api_key:
    print("Ooops: API key not found. Please set the OMDB_API_KEY environment variable.")
    exit(1)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Add production studio influence mapping
studio_influence = {
    "Warner Bros.": 10,
    "Paramount Pictures": 9,
    "Universal Pictures": 9,
    "20th Century Fox": 8,
    "Sony Pictures": 8,
    "New Line Cinema": 7,
    "Miramax Films": 7,
    "Lionsgate": 6,
    "DreamWorks Pictures": 8,
    "Columbia Pictures": 9,
}

director_popularity = {
    "Frank Darabont": 10,
    "Christopher Nolan": 9,
    "Quentin Tarantino": 8,
    "Steven Spielberg": 10,
    "Martin Scorsese": 9,
}

top_actors_list = ["Morgan Freeman", "Al Pacino", "Christian Bale", "Robert De Niro", "Leonardo DiCaprio"]

franchise_popularity = {
    "Marvel": 10,
    "Star Wars": 9,
    "Harry Potter": 8,
    "The Lord of the Rings": 9,
    "Fast & Furious": 7,
    "Batman": 9,
    "Spider-Man": 8,
    "James Bond": 8,
    "Transformers": 6,
    "The Avengers": 10,
}

# Function to assign production studio influence score
def get_studio_influence(studio):
    """Assign an influence score to the production studio."""
    try:
        return studio_influence.get(studio, 5)  # Default to 5 for unknown studios
    except Exception as e:
        print(f"Error in get_studio_influence: {e}")
        return 5

def extract_awards_count(awards):
    try:
        if isinstance(awards, str):
            numbers = [int(num) for num in re.findall(r'\d+', awards)]
            return sum(numbers)
        return 0
    except Exception as e:
        print(f"Error extracting awards count: {e}")
        return 0

# Function to calculate co-actor network strength
def co_actor_network_strength(actors, df):
    try:
        if isinstance(actors, str):
            actor_list = actors.split(', ')
            strength = 0
            for actor in actor_list:
                # Find co-actors in the dataset
                co_actors = df[df['Actors'].str.contains(actor, na=False, case=False)]['Actors']
                co_actor_list = [a for co in co_actors for a in co.split(', ') if a != actor]
                # Sum the number of movies co-actors have appeared in
                for co_actor in set(co_actor_list):
                    strength += df[df['Actors'].str.contains(co_actor, na=False, case=False)].shape[0]
            return strength
        return 0
    except Exception as e:
        print(f"Error calculating co-actor network strength: {e}")
        return 0

# Extract the number of awards won
def extract_awards_won(awards_str):
    try:
        awards_pattern = re.compile(r"(\d+)\s+win", re.IGNORECASE)
        match = awards_pattern.search(awards_str)
        return int(match.group(1)) if match else 0
    except Exception as e:
        print(f"Error extracting awards won: {e}")
        return 0

# Function to determine franchise impact
def get_franchise_impact(title):
    try:
        for franchise, score in franchise_popularity.items():
            if franchise.lower() in title.lower():
                return score
        return 5  # Default score for non-franchise movies
    except Exception as e:
        print(f"Error in get_franchise_impact: {e}")
        return 5

# Function to estimate social media buzz
def get_social_media_buzz(title):
    try:
        buzz_keywords = {
            "Avengers": 10,
            "Batman": 9,
            "Spider-Man": 8,
            "Frozen": 8,
            "Fast & Furious": 7,
            "Harry Potter": 9,
        }
        for keyword, score in buzz_keywords.items():
            if keyword.lower() in title.lower():
                return score
        return 5  # Default score for movies without significant buzz
    except Exception as e:
        print(f"Error in get_social_media_buzz: {e}")
        return 5

# New feature: Calculate critical reception sentiment
def get_critical_reception_sentiment(plot):
    try:
        if isinstance(plot, str):
            sentiment = analyzer.polarity_scores(plot)
            return sentiment['compound']
        return 0  # Default to neutral sentiment if plot is missing
    except Exception as e:
        print(f"Error in get_critical_reception_sentiment: {e}")
        return 0

# New feature: Categorize movies by runtime
def categorize_movie_length(runtime):
    try:
        if pd.notnull(runtime):
            if runtime < 90:
                return "Short"
            elif 90 <= runtime <= 150:
                return "Average"
            else:
                return "Long"
        return "Unknown"  # Handle missing or unknown runtime
    except Exception as e:
        print(f"Error categorizing movie length: {e}")
        return "Unknown"

# Fetch data for each movie
movie_data = []
for title in movie_titles:
    try:
        data = get_movie_data(title)
        if data:
            movie_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")

# Create DataFrame
df = pd.DataFrame(movie_data)

# Error handling: Fill missing data or replace with defaults
try:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)  # Replace missing years with the current year
    df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())  # Fill missing ratings with median
    df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')  # Convert to datetime
    df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)  # Handle missing years
except Exception as e:
    print(f"Error in data transformation: {e}")

# Handle missing Genres by assigning neutral sentiment
df['Genre'] = df['Genre'].fillna('Unknown')
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Handle missing Holiday Release information
df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release).fillna(False)

# Handle missing Rotten Tomatoes Scores by assuming '0%'
df['Rotten_Tomatoes_Score'] = df['Rotten_Tomatoes_Score'].fillna('0%')
df['RT_Sentiment'] = df['Rotten_Tomatoes_Score'].apply(get_rt_sentiment)

# Replace missing values for custom features
df['Awards_Won'] = df['Awards'].apply(extract_awards_won).fillna(0)
df['Co_Actor_Network_Strength'] = df['Actors'].apply(lambda x: co_actor_network_strength(x, df) if pd.notnull(x) else 0)
df['Studio_Influence'] = df['Production'].apply(get_studio_influence).fillna(5)
df['Top_Actors_Count'] = df['Actors'].apply(lambda x: sum([1 for actor in top_actors_list if pd.notnull(x) and actor in x]))

# Replace missing Director Popularity with default score
df['Director_Popularity'] = df['Director'].apply(lambda x: director_popularity.get(x, 5) if pd.notnull(x) else 5)

# Replace missing Budget to BoxOffice Ratio with default
df['Budget_to_BoxOffice_Ratio'] = df['Budget_to_BoxOffice_Ratio'].fillna(1.0)

# Add Franchise Impact feature
df['Franchise_Impact'] = df['Title'].apply(get_franchise_impact)

# Add Social Media Buzz feature
df['Social_Media_Buzz'] = df['Title'].apply(get_social_media_buzz)

# Add Critical Reception Sentiment feature
df['Critical_Reception_Sentiment'] = df['Plot'].apply(get_critical_reception_sentiment)

# Add Movie Length Category feature
df['Movie_Length_Category'] = df['Runtime_Minutes'].apply(categorize_movie_length)

# Drop rows where 'Rating' or critical features are missing
df = df.dropna(subset=['Rating', 'Year', 'Runtime_Minutes', 'Director', 'Actors'])

# Display the final cleaned DataFrame
print("DataFrame after cleaning and handling missing data:")
print(df.head())


