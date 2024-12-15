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

# Function to determine franchise impact
def get_franchise_impact(title):
    """
    Determine if the movie belongs to a popular franchise and assign a score.
    """
    for franchise, score in franchise_popularity.items():
        if franchise.lower() in title.lower():
            return score
    return 5  # Default score for non-franchise movies

# Function to estimate social media buzz
def get_social_media_buzz(title):
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

# New feature: Calculate Director's Previous Movie Performance
def get_director_previous_performance(director, df):
    """Calculate the average rating for the director's previous movies."""
    director_movies = df[df['Director'] == director]
    return director_movies['Rating'].mean() if not director_movies.empty else 5.0  # Default to 5 if no movies by this director

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

# Add Franchise Impact feature
df['Franchise_Impact'] = df['Title'].apply(get_franchise_impact)

# Add Social Media Buzz feature
df['Social_Media_Buzz'] = df['Title'].apply(get_social_media_buzz)

# Add Critical Reception Sentiment feature
df['Critical_Reception_Sentiment'] = df['Plot'].apply(get_critical_reception_sentiment)

# Add Movie Length Category feature
df['Movie_Length_Category'] = df['Runtime_Minutes'].apply(categorize_movie_length)

# Add Director's Previous Movie Performance feature
df['Director_Previous_Performance'] = df['Director'].apply(lambda x: get_director_previous_performance(x, df))

# Drop rows where 'Rating' or critical features are missing
df = df.dropna(subset=['Rating', 'Year', 'Runtime_Minutes', 'Director', 'Actors'])

# Display the final cleaned DataFrame
print("DataFrame after cleaning and handling missing data:")
print(df.head())

# Features for modeling (including the new feature)
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Runtime_Minutes', 
            'RT_Sentiment', 'Awards_Won', 'Top_Actors_Count', 'Movie_Age', 
            'Director_Popularity', 'Studio_Influence', 'Budget_to_BoxOffice_Ratio', 
            'Audience_Sentiment', 'Co_Actor_Network_Strength', 'Franchise_Impact', 
            'Social_Media_Buzz', 'Critical_Reception_Sentiment', 'Movie_Length_Category',
            'Director_Previous_Performance']  # New feature added here

X = pd.get_dummies(df[features], drop_first=True)  # Handle categorical variables
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
