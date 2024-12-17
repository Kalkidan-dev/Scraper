import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import re

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Your OMDb API key
api_key = '121c5367'

# Function to get movie data from OMDb with error handling
def get_movie_data(title):
    params = {'t': title, 'apikey': api_key}
    try:
        response = requests.get('http://www.omdbapi.com/', params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'True':
            return data
        else:
            print(f"Error: No data found for title '{title}' - Reason: {data.get('Error')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for title '{title}': {e}")
        return None

# Function to analyze the sentiment of movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']

# Function to convert BoxOffice to numeric
def convert_box_office_to_numeric(box_office):
    if isinstance(box_office, str) and box_office.startswith('$'):
        return int(box_office[1:].replace(',', ''))
    return 0

# Extract Awards Count
def extract_awards_count(awards):
    if isinstance(awards, str):
        numbers = [int(num) for num in re.findall(r'\d+', awards)]
        return sum(numbers)
    return 0

# Add Genre Diversity feature
def calculate_genre_diversity(genre):
    if isinstance(genre, str):
        genres = genre.split(',')
        unique_genres = set(genres)
        return len(unique_genres) / len(genres) if len(genres) > 0 else 0
    return 0

# Add Release Month Sentiment feature
def release_month_sentiment(release_date):
    if isinstance(release_date, str) and release_date:
        try:
            release_month = datetime.strptime(release_date, '%d %b %Y').month
            month_sentiment = {
                1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.8, 6: 0.9, 
                7: 1.0, 8: 0.9, 9: 0.4, 10: 0.6, 11: 0.7, 12: 0.5
            }
            return month_sentiment.get(release_month, 0.0)
        except ValueError:
            return 0.0
    return 0.0

# Add Actor Diversity feature
def calculate_actor_diversity(actors):
    if isinstance(actors, str):
        unique_actors = set(actors.split(', '))
        return len(unique_actors)
    return 0

# Add Weekend Release Indicator
def is_weekend_release(release_date):
    if isinstance(release_date, str) and release_date:
        try:
            release_day = datetime.strptime(release_date, '%d %b %Y').weekday()
            return 1 if release_day in [5, 6] else 0  # 5 = Saturday, 6 = Sunday
        except ValueError:
            return 0
    return 0

# Add Sequel Indicator
def is_sequel(title):
    sequels = ['2', 'II', 'III', 'IV', 'V']
    for sequel in sequels:
        if sequel in title:
            return 1
    return 0

# Director's Previous Success
def director_previous_success(director, df):
    if isinstance(director, str):
        director_movies = df[df['Director'] == director]
        return director_movies['BoxOffice'].sum()  # Sum of BoxOffice earnings of all the movies directed by the same director
    return 0

# Movie Popularity Trend
def movie_popularity_trend(row):
    if row['BoxOffice'] > 0 and row['Rating'] > 7.0:
        return 1  # Positive trend
    elif row['BoxOffice'] < 100000000 and row['Rating'] < 6.0:
        return 0  # Negative trend
    else:
        return 1 if row['Rating'] > 6.0 else 0

# Function to simulate social media buzz
def generate_social_media_buzz(title):
    """
    Simulate social media buzz as a random number.
    Replace this with real data from a social media API if available.
    """
    return random.randint(1000, 1000000)  # Simulate mentions between 1k and 1M

# New Feature: Count Main Actors' Popularity Score
def actors_popularity_score(actors):
    """
    Simulate actors' popularity based on predefined scores.
    Replace with real data if available.
    """
    predefined_popularity = {
        'Robert Downey Jr.': 95, 'Scarlett Johansson': 90, 'Leonardo DiCaprio': 92,
        'Chris Evans': 88, 'Brad Pitt': 85, 'Angelina Jolie': 87, 'Tom Cruise': 93
    }
    if isinstance(actors, str):
        actors_list = actors.split(', ')
        total_score = sum(predefined_popularity.get(actor, 50) for actor in actors_list)
        return total_score / len(actors_list)
    return 50

# New Feature: Director Experience
def director_experience(director, df):
    """
    Calculate the number of movies a director has directed before.
    """
    if isinstance(director, str):
        return len(df[df['Director'] == director])
    return 0

# Add all features from previous and new ones
features = [
    'Year', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity',
    'Genre_Sentiment', 'BoxOffice', 'Awards_Count', 'Genre_Diversity',
    'Release_Month_Sentiment', 'Weekend_Release', 'Sequel', 
    'Critic_Reviews_Sentiment', 'Audience_Engagement_Score', 'Social_Media_Buzz',
    'Actors_Popularity_Score', 'Director_Experience'
]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

# Define the target variable (e.g., 'Rating')
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RandomForestRegressor and GridSearchCV parameters
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearch
best_rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)

# Print evaluation metrics
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Visualize Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Regressor - Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
