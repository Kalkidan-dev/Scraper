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
    return studio_influence.get(studio, 5)  # Default to 5 for unknown studios

def extract_awards_count(awards):
    try:
        if isinstance(awards, str):
            numbers = [int(num) for num in re.findall(r'\d+', awards)]
            return sum(numbers)
        return 0
    except Exception as e:
        print(f"Error extracting awards count: {e}")
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

# New feature: Categorize movies by runtime
def categorize_movie_length(runtime):
    if pd.notnull(runtime):
        if runtime < 90:
            return "Short"
        elif 90 <= runtime <= 150:
            return "Average"
        else:
            return "Long"
    return "Unknown"

# Fetch data for each movie
movie_titles = ["Inception", "The Dark Knight", "Titanic"]  # Example movie list

def get_movie_data(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

movie_data = [get_movie_data(title) for title in movie_titles if get_movie_data(title)]

# Create DataFrame
df = pd.DataFrame(movie_data)

# Error handling: Fill missing data or replace with defaults
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)
df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')
df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)

def add_release_season(df, features):
    def classify_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Release_Month'] = df['Release_Date'].dt.month
    df['Release_Season'] = df['Release_Month'].apply(classify_season)
    df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)
    features += [col for col in df.columns if col.startswith('Release_Season_')]
    return df, features

features = []
df, features = add_release_season(df, features)

# Train the model
X = df[features]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Updated Mean Squared Error: {mse}')
print(f'Updated R-squared: {r2}')
