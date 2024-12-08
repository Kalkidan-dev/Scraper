import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
from datetime import datetime
import requests
import re
import random 

# Your OMDb API key
api_key = '121c5367'

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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

# Enhanced Actor Popularity
def enhanced_actor_popularity(actor_name):
    if isinstance(actor_name, str):
        lead_actor = actor_name.split(', ')[0]
        total_votes = 0
        total_rating = 0
        movie_count = 0
        actor_movies = get_actor_movies(lead_actor)  # Implement this function
        for movie_data in actor_movies:
            imdb_votes = movie_data.get('imdbVotes', '0').replace(',', '')
            try:
                total_votes += int(imdb_votes)
            except ValueError:
                total_votes += 0
            imdb_rating = movie_data.get('imdbRating', '0')
            try:
                total_rating += float(imdb_rating)
                movie_count += 1
            except ValueError:
                pass
        avg_rating = total_rating / movie_count if movie_count > 0 else 0
        popularity_score = (total_votes / 1_000) + (avg_rating * 10)
        return round(popularity_score, 2)
    return 0

def get_actor_movies(actor_name):
    api_key = "your_api_key_here"
    movies = []
    for i in range(1, 3):  # Adjust pagination as needed
        response = requests.get(f"http://www.omdbapi.com/?apikey={api_key}&s={actor_name}&type=movie&page={i}")
        if response.status_code == 200:
            data = response.json()
            if "Search" in data:
                for movie in data["Search"]:
                    movie_data = get_movie_data(movie['Title'])
                    if movie_data:
                        movies.append(movie_data)
            else:
                break
        else:
            break
    return movies
def extract_audience_review_count(imdb_votes):
    """
    Extract audience review count from the imdbVotes column.
    If imdbVotes is not a valid number, return 0.
    """
    if isinstance(imdb_votes, str):
        try:
            return int(imdb_votes.replace(',', ''))
        except ValueError:
            return 0
    elif isinstance(imdb_votes, (int, float)):
        return int(imdb_votes)
    return 0  # Default for missing or invalid data
 # Import random module for simulation

# Add Social Media Mentions feature
def generate_social_media_mentions(title):
    """
    Generate a simulated value for social media mentions.
    In a real-world scenario, integrate with a social media API.
    """
    if isinstance(title, str):
        # Simulate social media mentions with a random number between 1000 and 100000
        return random.randint(1000, 100000)
    return 0
# Fetch data for each movie
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:
        movie_data.append(data)

if not movie_data:
    print("No movie data was retrieved. Exiting...")
    exit()

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 
                    'imdbVotes', 'BoxOffice', 'Awards', 'Released', 'Actors']
df = df[required_columns]

# Adding Social Media Mentions to the DataFrame
df['Social_Media_Mentions'] = df['Title'].apply(generate_social_media_mentions)

# Include the feature in the feature set
features.append('Social_Media_Mentions')
# Extract Rotten Tomatoes and Metascore ratings
def extract_ratings(ratings, source):
    for rating in ratings:
        if rating['Source'] == source:
            return rating['Value']
    return None

# Add Rotten Tomatoes and Metascore to DataFrame
df['Rotten Tomatoes'] = df['Ratings'].apply(lambda x: extract_ratings(x, 'Rotten Tomatoes'))
df['Metascore'] = df['Metascore'].astype(float)

# New Feature: Critic Reviews Sentiment
def fetch_critic_reviews_sentiment(title):
    params = {'t': title, 'apikey': api_key}
    try:
        response = requests.get('http://www.omdbapi.com/', params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'True' and 'Ratings' in data:
            reviews = [rating['Value'] for rating in data['Ratings'] if 'Metascore' in rating or 'Rotten Tomatoes' in rating]
            if reviews:
                sentiment_scores = [analyzer.polarity_scores(review)['compound'] for review in reviews]
                return np.mean(sentiment_scores)
        return 0.0
    except requests.exceptions.RequestException as e:
        print(f"Request error for title '{title}': {e}")
        return 0.0

# Example list of movie titles
movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
    '12 Angry Men', 'Schindler\'s List', 'Pulp Fiction',
    'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
    'Fight Club', 'Forrest Gump'
]

# Example budget data (in millions)
budget_data = {
    'The Shawshank Redemption': 25, 'The Godfather': 6, 'The Dark Knight': 185,
    '12 Angry Men': 0.35, 'Schindler\'s List': 22, 'Pulp Fiction': 8,
    'The Lord of the Rings: The Return of the King': 94, 'The Good, the Bad and the Ugly': 1.2,
    'Fight Club': 63, 'Forrest Gump': 55
}

# Fetch data for each movie
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:
        movie_data.append(data)

if not movie_data:
    print("No movie data was retrieved. Exiting...")
    exit()

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 'imdbVotes', 'BoxOffice', 'Awards', 'Released', 'Actors']
df = df[required_columns]
df['Rating'] = df['imdbRating'].astype(float)

# Feature Engineering
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)
df['Year'] = df['Year'].astype(int)
df['Director_Popularity'] = df['Director'].map(df['Director'].value_counts())
df['Runtime'] = df['Runtime'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else np.nan)
df['Budget'] = df['Title'].map(budget_data)
df['Movie_Popularity'] = df['imdbVotes'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else 0)
df['BoxOffice'] = df['BoxOffice'].apply(convert_box_office_to_numeric)
df['Awards_Count'] = df['Awards'].apply(extract_awards_count)
df['Genre_Diversity'] = df['Genre'].apply(calculate_genre_diversity)
df['Release_Month_Sentiment'] = df['Released'].apply(release_month_sentiment)
df['Weekend_Release'] = df['Released'].apply(is_weekend_release)
df['Sequel'] = df['Title'].apply(is_sequel)
df['Critic_Reviews_Sentiment'] = df['Title'].apply(fetch_critic_reviews_sentiment)

# Prepare Features and Target
features = ['Year', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity',
            'Genre_Sentiment', 'BoxOffice', 'Awards_Count', 'Genre_Diversity',
            'Release_Month_Sentiment', 'Weekend_Release', 'Sequel', 'Critic_Reviews_Sentiment']
X = df[features]
y = df['Rating']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
