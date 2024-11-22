import requests
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime
import re

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

# Function to analyze the sentiment of the movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']  # Use the compound score for overall sentiment

# Function to convert BoxOffice to numeric
def convert_box_office_to_numeric(box_office):
    if isinstance(box_office, str) and box_office.startswith('$'):
        return int(box_office[1:].replace(',', ''))
    return 0

# Extracting Awards Count
def extract_awards_count(awards):
    if isinstance(awards, str):
        # Extract numbers in the format "Won X awards" or "Nominated for Y awards"
        numbers = [int(num) for num in re.findall(r'\d+', awards)]
        return sum(numbers)  # Sum all numbers extracted
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
            # Extract the month from the release date
            release_month = datetime.strptime(release_date, '%d %b %Y').month
            # Define sentiment scores for months (arbitrary example scores)
            month_sentiment = {
                1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.8, 6: 0.9, 
                7: 1.0, 8: 0.9, 9: 0.4, 10: 0.6, 11: 0.7, 12: 0.5
            }
            return month_sentiment.get(release_month, 0.0)
        except ValueError:
            return 0.0
    return 0.0

# Add IMDb Rating per Director
def calculate_imdb_rating_per_director(director, ratings_df):
    # Check if the director exists in the DataFrame and calculate the average IMDb rating
    director_movies = ratings_df[ratings_df['Director'] == director]
    if not director_movies.empty:
        return director_movies['Rating'].mean()  # Average IMDb rating for that director
    return np.nan

# Example list of top-rated movie titles to fetch
movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
    '12 Angry Men', 'Schindler\'s List', 'Pulp Fiction',
    'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
    'Fight Club', 'Forrest Gump'
]

# Example budget data (in millions) for the movies
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
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 'imdbVotes', 'BoxOffice', 'Awards', 'Released']
df = df[required_columns]
df['Rating'] = df['imdbRating'].astype(float)

# Analyze genre sentiment
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Convert Year to numeric
df['Year'] = df['Year'].astype(int)

# Calculate Director Popularity
df['Director_Popularity'] = df['Director'].map(df['Director'].value_counts())

# Convert Runtime from "X min" format to numeric
df['Runtime'] = df['Runtime'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else np.nan)

# Add Budget data
df['Budget'] = df['Title'].map(budget_data)

# Add Movie Popularity based on IMDb Votes
df['Movie_Popularity'] = df['imdbVotes'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else 0)

# Add Number of Genres
df['Num_Genres'] = df['Genre'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Add Rating per Genre
df['Rating_per_Genre'] = df.apply(
    lambda row: row['Rating'] / row['Num_Genres'] if row['Num_Genres'] > 0 else 0, axis=1
)

# Add Movie Age
current_year = datetime.now().year
df['Movie_Age'] = current_year - df['Year']

# Add Box Office Revenue per Genre
df['BoxOffice'] = df['BoxOffice'].apply(convert_box_office_to_numeric)
df['BoxOffice_per_Genre'] = df.apply(
    lambda row: row['BoxOffice'] / row['Num_Genres'] if row['Num_Genres'] > 0 else 0, axis=1
)

# Add Awards Count
df['Awards_Count'] = df['Awards'].apply(extract_awards_count)

# Add Genre Diversity
df['Genre_Diversity'] = df['Genre'].apply(calculate_genre_diversity)

# Add Release Month Sentiment
df['Release_Month_Sentiment'] = df['Released'].apply(release_month_sentiment)

# Add IMDb Rating per Director
df['IMDb_Rating_per_Director'] = df['Director'].apply(lambda director: calculate_imdb_rating_per_director(director, df))

# Features for the model
features = [
    'Year', 'Genre_Sentiment', 'Director_Popularity', 'Runtime', 
    'Budget', 'Movie_Popularity', 'Num_Genres', 'Rating_per_Genre', 
    'Movie_Age', 'BoxOffice_per_Genre', 'Awards_Count', 'Genre_Diversity',
    'Release_Month_Sentiment', 'IMDb_Rating_per_Director'
]

# X = feature set
X = df[features]

# y = target variable (IMDb Rating)
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Scatter plot: IMDb Rating vs Release Month Sentiment
plt.figure(figsize=(10, 6))
plt.scatter(df['Rating'], df['Release_Month_Sentiment'], alpha=0.5, color='blue')
plt.xlabel('IMDb Rating')
plt.ylabel('Release Month Sentiment')
plt.title('IMDb Rating vs Release Month Sentiment')
plt.show()
