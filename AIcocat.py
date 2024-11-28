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

# Function to assign production studio influence score
def get_studio_influence(studio):
    """Assign an influence score to the production studio."""
    return studio_influence.get(studio, 5)  # Default to 5 for unknown studios

# Extract the number of awards won
def extract_awards_won(awards_str):
    awards_pattern = re.compile(r"(\d+)\s+win", re.IGNORECASE)
    match = awards_pattern.search(awards_str)
    return int(match.group(1)) if match else 0

# Count top actors in the movie
def count_top_actors(actors_str):
    return sum(1 for actor in top_actors_list if actor in actors_str)

# Function to get movie data from OMDb with retry logic
def get_movie_data(title, retries=3, delay=5):
    params = {'t': title, 'apikey': api_key}
    for attempt in range(retries):
        try:
            response = requests.get('http://www.omdbapi.com/', params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('Response') == 'True':
                runtime_str = data.get('Runtime', '0 min')
                runtime = int(runtime_str.split()[0]) if runtime_str.split()[0].isdigit() else 0
                data['Runtime_Minutes'] = runtime

                # Extract the number of awards won
                awards_str = data.get('Awards', '')
                data['Awards_Won'] = extract_awards_won(awards_str)

                # Count top actors
                actors_str = data.get('Actors', '')
                data['Top_Actors_Count'] = count_top_actors(actors_str)

                # Get director popularity
                director_name = data.get('Director', '')
                data['Director_Popularity'] = director_popularity.get(director_name, 5)  # Default to 5 if not listed

                # Add studio influence
                studio_name = data.get('Production', '')
                data['Studio_Influence'] = get_studio_influence(studio_name)

                return data
            else:
                print(f"Error: No data found for title '{title}' - {data.get('Error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Request error for title '{title}': {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(delay)
            else:
                print("Failed to fetch data after multiple attempts.")
                return None

# Function to analyze the sentiment of the movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']

# Function to create a holiday release indicator
def is_holiday_release(date):
    holiday_dates = ['12-25', '11-26']
    return int(date.strftime('%m-%d') in holiday_dates)

# Convert Rotten Tomatoes score to sentiment
def get_rt_sentiment(rt_score_str):
    try:
        rt_score = int(rt_score_str.strip('%'))
        if rt_score >= 75:
            return 1  # Positive sentiment
        elif rt_score >= 50:
            return 0  # Neutral sentiment
        else:
            return -1  # Negative sentiment
    except (ValueError, TypeError):
        return 0

# Example Rotten Tomatoes scores for the movie titles
rotten_tomatoes_scores = {
    'The Shawshank Redemption': '91%',
    'The Godfather': '98%',
    'The Dark Knight': '94%',
    '12 Angry Men': '100%',
    'Schindler\'s List': '98%',
    'Pulp Fiction': '92%',
    'The Lord of the Rings: The Return of the King': '93%',
    'The Good, the Bad and the Ugly': '97%',
    'Fight Club': '79%',
    'Forrest Gump': '71%'
}

# List of top-rated movie titles to fetch as an example
movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
    '12 Angry Men', 'Schindler\'s List', 'Pulp Fiction',
    'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
    'Fight Club', 'Forrest Gump'
]

# Fetch data for each movie
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:
        movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Clean and engineer features
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Rating'] = df['imdbRating'].astype(float)
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')
df['Movie_Age'] = datetime.now().year - df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)
df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release)
df['Rotten_Tomatoes_Score'] = df['Title'].map(rotten_tomatoes_scores).fillna('0%')
df['RT_Sentiment'] = df['Rotten_Tomatoes_Score'].apply(get_rt_sentiment)

# Features for modeling
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Runtime_Minutes', 
            'RT_Sentiment', 'Awards_Won', 'Top_Actors_Count', 'Movie_Age', 
            'Director_Popularity', 'Studio_Influence']
X = df[features]
y = df['Rating'].astype(float)

# Train-test split and modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot predictions
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Movie Ratings')
plt.grid(True)
plt.show()
