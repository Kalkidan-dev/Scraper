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
    if isinstance(awards, str):
        numbers = [int(num) for num in re.findall(r'\d+', awards)]
        return sum(numbers)
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
    try:
        awards_pattern = re.compile(r"(\d+)\s+win", re.IGNORECASE)
        match = awards_pattern.search(awards_str)
        return int(match.group(1)) if match else 0
    except Exception as e:
        print(f"Error extracting awards won: {e}")
        return 0
    awards_pattern = re.compile(r"(\d+)\s+win", re.IGNORECASE)
    match = awards_pattern.search(awards_str)
    return int(match.group(1)) if match else 0

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
    """
    Determine if the movie belongs to a popular franchise and assign a score.
    """
    for franchise, score in franchise_popularity.items():
        if franchise.lower() in title.lower():
            return score
    return 5  # Default score for non-franchise movies

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
    """
    Simulate social media buzz score based on movie title.
    Higher scores represent more buzz (e.g., mentions, hashtags).
    """
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
    """
    Use VADER sentiment analysis to calculate the sentiment of the plot.
    """
    if isinstance(plot, str):
        sentiment = analyzer.polarity_scores(plot)
        return sentiment['compound']
    return 0  # Default to neutral sentiment if plot is missing

# New feature: Categorize movies by runtime
def categorize_movie_length(runtime):
    """
    Categorize movies into 'Short', 'Average', and 'Long' based on runtime.
    """
    if pd.notnull(runtime):
        if runtime < 90:
            return "Short"
        elif 90 <= runtime <= 150:
            return "Average"
        else:
            return "Long"
    return "Unknown"  # Handle missing or unknown runtime
# New feature: Director's Historical Success Rate
def get_director_success_rate(director, df):
    try:
        if pd.notnull(runtime):
            if runtime < 90:
                return "Short"
            elif 90 <= runtime <= 150:
                return "Average"
            else:
                return "Long"
        return "Unknown"  # Handle missing or unknown runtime
        # Filter the movies directed by the same director
        director_movies = df[df['Director'] == director]
        
        if not director_movies.empty:
            # Calculate the average IMDb rating and average Box Office revenue
            avg_imdb_rating = director_movies['imdbRating'].mean()
            avg_box_office = director_movies['BoxOffice'].mean()
            
            # A simple formula to calculate success rate, giving more weight to IMDb ratings
            success_rate = (avg_imdb_rating * 0.7) + (avg_box_office * 0.3)  # Weight the IMDb higher
            return success_rate
        return 5  # Default success rate if there are no movies by this director
    except Exception as e:
        print(f"Error categorizing movie length: {e}")
        return "Unknown"
        print(f"Error calculating success rate for {director}: {e}")
        return 5  # Default score if an error occurs

# Fetch data for each movie
# Fetch data for each movie (your method for obtaining movie data should go here)
movie_data = []
for title in movie_titles:
    try:
        data = get_movie_data(title)
        if data:
            movie_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
for title in movie_titles:  # Assuming 'movie_titles' is already defined elsewhere
    data = get_movie_data(title)  # Your function for fetching movie data
    if data:
        movie_data.append(data)

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
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)  # Replace missing years with the current year
df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())  # Fill missing ratings with median
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')  # Convert to datetime
df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)  # Handle missing years

   def add_release_season(df, features):
       """
       Add a new feature for the release season of the movie.

       Args:
           df (pd.DataFrame): The input DataFrame.
           features (list): The list of feature column names.

       Returns:
           pd.DataFrame, list: Updated DataFrame and features list.
       """
       def classify_season(month):
           """Classify the movie's release month into a season."""
           if month in [12, 1, 2]:
               return 'Winter'
           elif month in [3, 4, 5]:
               return 'Spring'
           elif month in [6, 7, 8]:
               return 'Summer'
           else:
               return 'Fall'

       # Assuming the DataFrame has a 'Release_Month' column
       df['Release_Season'] = df['Release_Month'].apply(classify_season)

       # One-hot encode the 'Release_Season' feature
       df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)

       # Update the features list to include the new 'Release_Season' columns
       features += [col for col in df.columns if col.startswith('Release_Season_')]

       return df, features

   # Call the new function without modifying the existing code
   df, features = add_release_season(df, features)

   # Re-train the model with the updated features
   X = df[features]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = LinearRegression()
   model.fit(X_train, y_train)

   # Recalculate predictions and metrics
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)

   print(f'Updated Mean Squared Error: {mse}')
   print(f'Updated R-squared: {r2}')

   # Example prediction with the new feature
   predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100)
   print(f'Predicted Rating for a movie in 2024: {predicted_rating:.2f}')
   
