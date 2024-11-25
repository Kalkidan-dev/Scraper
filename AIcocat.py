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

# Load OMDb API key from environment variable
api_key = os.getenv("OMDB_API_KEY")

if not api_key:
    print("Ooops: API key not found. Please set the OMDB_API_KEY environment variable.")
    exit(1)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

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
                try:
                    runtime = int(runtime_str.split()[0])
                except ValueError:
                    runtime = 0
                data['Runtime_Minutes'] = runtime
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

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Released', 'Runtime_Minutes']]
df['Rating'] = df['imdbRating'].astype(float)

# Convert 'Released' column to datetime
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')

# Create Day of the Week feature
df['Day_of_Week'] = df['Release_Date'].dt.day_name()

# Create binary feature for weekend vs. weekday
df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

# One-hot encode the day of the week
df = pd.get_dummies(df, columns=['Day_of_Week'], drop_first=True)

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Apply the holiday release indicator
df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release)

# Add a Director Popularity Score
director_popularity = {
    'Frank Darabont': 9,
    'Francis Ford Coppola': 10,
    'Christopher Nolan': 10,
    'Sidney Lumet': 8,
    'Steven Spielberg': 10,
    'Quentin Tarantino': 9,
    'Peter Jackson': 10,
    'Sergio Leone': 9,
    'David Fincher': 8,
    'Robert Zemeckis': 8
}
df['Director_Popularity'] = df['Director'].map(director_popularity).fillna(5)

# Add the "Budget" feature
budget_data = {
    'The Shawshank Redemption': 25,
    'The Godfather': 6,
    'The Dark Knight': 185,
    '12 Angry Men': 0.35,
    'Schindler\'s List': 22,
    'Pulp Fiction': 8,
    'The Lord of the Rings: The Return of the King': 94,
    'The Good, the Bad and the Ugly': 1.2,
    'Fight Club': 63,
    'Forrest Gump': 55
}
df['Budget'] = df['Title'].map(budget_data).fillna(10)

# Add the "Number of Awards" feature
awards_data = {
    'The Shawshank Redemption': 7,
    'The Godfather': 3,
    'The Dark Knight': 8,
    '12 Angry Men': 2,
    'Schindler\'s List': 12,
    'Pulp Fiction': 7,
    'The Lord of the Rings: The Return of the King': 17,
    'The Good, the Bad and the Ugly': 4,
    'Fight Club': 1,
    'Forrest Gump': 9
}
df['Number_of_Awards'] = df['Title'].map(awards_data).fillna(0)

# Add a new feature: Director Age at Release
director_birth_years = {
    'Frank Darabont': 1959,
    'Francis Ford Coppola': 1939,
    'Christopher Nolan': 1970,
    'Sidney Lumet': 1924,
    'Steven Spielberg': 1946,
    'Quentin Tarantino': 1963,
    'Peter Jackson': 1961,
    'Sergio Leone': 1929,
    'David Fincher': 1962,
    'Robert Zemeckis': 1952
}
df['Director_Birth_Year'] = df['Director'].map(director_birth_years)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Director_Birth_Year'] = pd.to_numeric(df['Director_Birth_Year'], errors='coerce')

# Fill NaN values in Director_Birth_Year
df['Director_Birth_Year'].fillna(0, inplace=True)

# Calculate the Director's Age at Release
df['Director_Age_At_Release'] = df['Year'] - df['Director_Birth_Year']

# Add the "Average IMDb Rating of the Cast" feature
cast_avg_rating = {
    'The Shawshank Redemption': 8.7,
    'The Godfather': 9.1,
    'The Dark Knight': 8.9,
    '12 Angry Men': 8.5,
    'Schindler\'s List': 8.8,
    'Pulp Fiction': 8.6,
    'The Lord of the Rings: The Return of the King': 8.9,
    'The Good, the Bad and the Ugly': 8.5,
    'Fight Club': 8.7,
    'Forrest Gump': 8.5
}
df['Avg_IMDb_Rating_of_Cast'] = df['Title'].map(cast_avg_rating).fillna(8.0)

# Ensure all relevant features are selected
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Is_Weekend', 
            'Runtime_Minutes', 'Director_Popularity', 'Budget', 
            'Number_of_Awards', 'Director_Age_At_Release', 'Avg_IMDb_Rating_of_Cast']

# Extract features
X = df[features]
y = df['Rating'].astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Display the actual vs predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Prediction function
def predict_rating(year, genre_sentiment, holiday_release, is_weekend, runtime, 
                   director_popularity, budget, number_of_awards, director_age, avg_cast_rating):
    features = [
        year, genre_sentiment, holiday_release, is_weekend, runtime, 
        director_popularity, budget, number_of_awards, director_age, avg_cast_rating
    ]
    return model.predict(np.array([features]))[0]

# Example prediction
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100, 10, 54, 8.5)
print(f'Predicted Rating for a movie in 2024: {predicted_rating}')
