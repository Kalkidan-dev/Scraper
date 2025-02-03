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

# New feature: Extract Oscar nominations
def extract_oscar_nominations(awards):
    try:
        if isinstance(awards, str):
            match = re.search(r"(\d+)\s+nomination", awards, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        return 0
    except Exception as e:
        print(f"Error extracting Oscar nominations: {e}")
        return 0

# Function to assign production studio influence score
def get_studio_influence(studio):
    """Assign an influence score to the production studio."""
    try:
        return studio_influence.get(studio, 5)  # Default to 5 for unknown studios
    except Exception as e:
        print(f"Error in get_studio_influence: {e}")
        return 5

# Function to categorize movies by runtime
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

# Fetch data for each movie
movie_titles = ["Inception", "The Dark Knight", "Titanic"]  # Replace with actual movie titles to fetch
movie_data = []
for title in movie_titles:
    try:
        data = get_movie_data(title)  # Replace with actual data fetching function
        if data:
            movie_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")

# Create DataFrame
df = pd.DataFrame(movie_data)

# Apply new feature
df['Oscar_Nominations'] = df['Awards'].apply(extract_oscar_nominations)

# Error handling: Fill missing data or replace with defaults
try:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)  # Replace missing years with the current year
    df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())  # Fill missing ratings with median
    df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')  # Convert to datetime
    df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)  # Handle missing years
except Exception as e:
    print(f"Error in data transformation: {e}")

def genre_familiarity_index(director, genre, df):
    """
    Calculate how familiar a director is with the genres of the current movie.
    """
    if isinstance(director, str) and isinstance(genre, str):
        # Split the genres into a list
        genres = genre.split(', ')
        # Filter the dataset for movies by this director
        director_movies = df[df['Director'] == director]
        # Count the occurrence of each genre in the director's previous movies
        genre_count = sum(director_movies['Genre'].str.contains(g, case=False, na=False).sum() for g in genres)
        # Normalize by the number of genres in the current movie
        return genre_count / len(genres) if len(genres) > 0 else 0
    return 0

# New Feature: International Release Indicator
def international_release_indicator(country):
    """
    Check if a movie was released in multiple countries.
    """
    if isinstance(country, str):
        countries = country.split(', ')
        return 1 if len(countries) > 1 else 0
    return 0

def box_office_success(budget, box_office):
    """
    Determine if a movie was a box office success based on its budget and earnings.
    Returns 1 if box office earnings are greater than the budget, else 0.
    """
    if budget > 0 and box_office > 0:
        return 1 if box_office > budget else 0
    return 0  # If budget or box office is missing, assume failure (0)


# New Feature: Critics vs Audience Rating Disparity
def critics_vs_audience_disparity(critics_rating, audience_rating):
    """
    Calculate the absolute difference between critics' and audience ratings.
    """
    if critics_rating is not None and audience_rating is not None:
        return abs(critics_rating - audience_rating)
    return 0.0

# New Feature: Budget-to-BoxOffice Ratio
def budget_to_boxoffice_ratio(budget, box_office):
    """
    Calculate the ratio of budget to box office earnings.
    """
    if budget > 0 and box_office > 0:
        return budget / box_office
    return 0.0

def franchise_indicator(title):
    """
    Check if a movie title contains keywords commonly associated with franchises.
    """
    franchise_keywords = ['Marvel', 'Avengers', 'Star Wars', 'Harry Potter', 
                          'Fast & Furious', 'Mission Impossible', 'Transformers', 
                          'Spider-Man', 'Batman', 'Superman', 'Jurassic']
    for keyword in franchise_keywords:
        if keyword.lower() in title.lower():
            return 1
    return 0

def director_collaboration_frequency(director, actors, df):
    """
    Calculate how often a director collaborates with the same actors.
    """
    if isinstance(director, str) and isinstance(actors, str):
        # Split the actors into a list
        actor_list = actors.split(', ')
        
        # Filter the dataset for movies by the same director
        director_movies = df[df['Director'] == director]
        
        # Count collaborations with the same actors
        collaboration_count = 0
        for actor in actor_list:
            collaboration_count += director_movies['Actors'].apply(lambda x: actor in x if isinstance(x, str) else False).sum()
        
        return collaboration_count / len(actor_list) if len(actor_list) > 0 else 0
    return 0

def seasonal_popularity_score(release_date):
    """
    Assign a seasonal popularity score based on the release date.
    """
    if isinstance(release_date, str) and release_date:
        try:
            # Parse the release date to extract the month
            release_month = datetime.strptime(release_date, '%d %b %Y').month
            
            # Seasonal popularity scores (values can be adjusted based on real data)
            season_scores = {
                'Winter': 0.8,  # December, January, February
                'Spring': 0.7,  # March, April, May
                'Summer': 1.0,  # June, July, August
                'Fall': 0.6     # September, October, November
            }
            
            # Assign seasons based on the month
            if release_month in [12, 1, 2]:
                return season_scores['Winter']
            elif release_month in [3, 4, 5]:
                return season_scores['Spring']
            elif release_month in [6, 7, 8]:
                return season_scores['Summer']
            elif release_month in [9, 10, 11]:
                return season_scores['Fall']
        except ValueError:
            return 0.0
        return 0.0

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

    df['Release_Season'] = df['Release_Month'].apply(classify_season)
    df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)
    features += [col for col in df.columns if col.startswith('Release_Season_')]
    return df, features

# Call the new function
df, features = add_release_season(df, features)
df['Box_Office_Success'] = df.apply(lambda row: box_office_success(row['Budget'], row['BoxOffice']), axis=1)
features.append('Box_Office_Success')  # Add it to the features list


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

# Print the DataFrame to verify
print(df[['Title', 'Awards', 'Oscar_Nominations']])
