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

# Function to calculate co-actor network strength
def co_actor_network_strength(actors, df):
    """
    Calculate the co-actor network strength by summing the number of movies 
    their co-actors have appeared in within the dataset.
    """
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

# Extract the number of awards won
def extract_awards_won(awards_str):
    awards_pattern = re.compile(r"(\d+)\s+win", re.IGNORECASE)
    match = awards_pattern.search(awards_str)
    return int(match.group(1)) if match else 0

# (Remaining previously defined functions like `extract_golden_globe_wins`, 
# `calculate_budget_to_boxoffice_ratio`, `analyze_genre_sentiment` remain unchanged)

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

# Add Co-Actor Network Strength feature
df['Co_Actor_Network_Strength'] = df['Actors'].apply(lambda x: co_actor_network_strength(x, df))

# Features for modeling (including the new feature)
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Runtime_Minutes', 
            'RT_Sentiment', 'Awards_Won', 'Top_Actors_Count', 'Movie_Age', 
            'Director_Popularity', 'Studio_Influence', 'Budget_to_BoxOffice_Ratio', 
            'Audience_Sentiment', 'Co_Actor_Network_Strength']
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
