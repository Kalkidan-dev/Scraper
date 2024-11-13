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
    """Fetch movie data with retries in case of connection errors."""
    params = {'t': title, 'apikey': api_key}
    
    for attempt in range(retries):
        try:
            response = requests.get('http://www.omdbapi.com/', params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get('Response') == 'True':
                # Convert runtime to minutes if available
                runtime_str = data.get('Runtime', '0 min')
                try:
                    runtime = int(runtime_str.split()[0])  # Extract number of minutes
                except ValueError:
                    runtime = 0  # Default to 0 if conversion fails
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
    holiday_dates = ['12-25', '11-26']  # List of specific holiday dates (Christmas, example Thanksgiving)
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

# Prepare the data for prediction
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Features for the model, including Runtime
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Is_Weekend', 'Runtime_Minutes']
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

# Example of predicting the rating for a new movie
def predict_rating(year, genre_sentiment, holiday_release, is_weekend, runtime):
    return model.predict(np.array([[year, genre_sentiment, holiday_release, is_weekend, runtime]]))[0]

# Example
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120)
print(f'Predicted Rating for a movie in 2024: {predicted_rating:.2f}')

# Merge with Rotten Tomatoes data for analysis
rt_data = {
    'Title': movie_titles,
    'RT_Rating': [91, 98, 94, 100, 97, 92, 95, 97, 79, 71]
}
rt_df = pd.DataFrame(rt_data)
combined_df = pd.merge(df, rt_df, on='Title', how='inner')

# Correlation matrix
correlation_matrix = combined_df[['Rating', 'RT_Rating', 'Genre_Sentiment', 'Is_Holiday_Release', 'Is_Weekend']].astype(float).corr()
print(correlation_matrix)

# Scatter plots for analysis
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['Rating'], combined_df['RT_Rating'], alpha=0.5)
plt.xlabel('IMDb Rating')
plt.ylabel('Rotten Tomatoes Rating')
plt.title('Comparison of IMDb and Rotten Tomatoes Ratings')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(combined_df['Rating'], combined_df['Genre_Sentiment'], alpha=0.5, color='green')
plt.xlabel('IMDb Rating')
plt.ylabel('Genre Sentiment')
plt.title('IMDb Rating vs Genre Sentiment')
plt.show()

# Top 10 movies by rating
top_10_movies = df.sort_values(by='Rating', ascending=False).head(10)
print(top_10_movies)

# Histogram of IMDb ratings
plt.hist(df['Rating'], bins=20, edgecolor='k')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of IMDb Ratings')
plt.show()

# Bar plot of top 10 movies
plt.figure(figsize=(10, 6))
plt.barh(top_10_movies['Title'], top_10_movies['Rating'], color='skyblue')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.title('Top 10 IMDb Movies by Rating')
plt.gca().invert_yaxis()
plt.show()
