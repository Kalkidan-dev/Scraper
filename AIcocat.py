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

def extract_awards_count(awards):
    if isinstance(awards, str):
        numbers = [int(num) for num in re.findall(r'\d+', awards)]
        return sum(numbers)
    return 0

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

# New Feature: Calculate average sentiment score of movie reviews
def calculate_review_sentiment(reviews):
    """
    Calculate the average sentiment score from a list of reviews using VADER sentiment analysis.
    """
    if isinstance(reviews, list):
        sentiments = [analyzer.polarity_scores(review)['compound'] for review in reviews]
        return np.mean(sentiments) if sentiments else 0
    return 0

# Fetch data for each movie
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:
        movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Error handling: Fill missing data or replace with defaults
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)  # Replace missing years with the current year
df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())  # Fill missing ratings with median
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')  # Convert to datetime
df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)  # Handle missing years

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

# Add new feature: Review Sentiment
if 'Reviews' in df.columns:
    df['Review_Sentiment'] = df['Reviews'].apply(calculate_review_sentiment).fillna(0)
else:
    df['Review_Sentiment'] = 0

# Drop rows where 'Rating' or critical features are missing
df = df.dropna(subset=['Rating', 'Year', 'Runtime_Minutes', 'Director', 'Actors'])

# Display the final cleaned DataFrame
print("DataFrame after cleaning and handling missing data:")
print(df.head())

# Features for modeling (including the new feature)
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Runtime_Minutes', 
            'RT_Sentiment', 'Awards_Won', 'Top_Actors_Count', 'Movie_Age', 
            'Director_Popularity', 'Studio_Influence', 'Budget_to_BoxOffice_Ratio', 
            'Audience_Sentiment', 'Co_Actor_Network_Strength', 'Review_Sentiment']
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