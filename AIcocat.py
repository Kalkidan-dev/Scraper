import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os  # To access environment variables
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
    params = {
        't': title,
        'apikey': api_key
    }
    
    for attempt in range(retries):
        try:
            response = requests.get('http://www.omdbapi.com/', params=params)
            response.raise_for_status()  # Raise an error for bad status codes (like 404 or 500)
            
            data = response.json()
            
            if data.get('Response') == 'True':
                return data
            else:
                print(f"Error: No data found for title '{title}' - {data.get('Error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Request error for title '{title}': {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(delay)  # Wait before retrying
            else:
                print("Failed to fetch data after multiple attempts.")
                return None

# Function to analyze the sentiment of the movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']  # Use the compound score for overall sentiment

# Function to create a holiday release indicator
def is_holiday_release(date):
    # List of specific holiday dates
    holiday_dates = [
        '12-25',  # Christmas
        '11-26',  # Example: Thanksgiving (date changes yearly, consider using date ranges)
        # Add more holiday dates or use a more dynamic method
    ]
    return int(date.strftime('%m-%d') in holiday_dates)

# List of top-rated movie titles to fetch as Example
movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
    '12 Angry Men', 'Schindler\'s List', 'Pulp Fiction',
    'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
    'Fight Club', 'Forrest Gump'
]

# Fetch data for each movie with error handling and retry logic
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:  # Only append if data was successfully retrieved
        movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Released']]  # Added 'Released' column
df['Rating'] = df['imdbRating'].astype(float)

# Convert 'Released' column to datetime
df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Apply the holiday release indicator
df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release)

# Step 2: Prepare the data for prediction
# Convert Year to a numeric feature and ensure Genre_Sentiment is float
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Features we will use to predict IMDb Rating
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release']

# X = feature set (Year, Genre_Sentiment, Is_Holiday_Release)
X = df[features]

# y = target variable (IMDb Rating)
y = df['Rating'].astype(float)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')  # R-squared tells us how well the model explains the variance in the data

# Display the actual vs predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Make a prediction for a new movie
def predict_rating(year, genre_sentiment, holiday_release):
    return model.predict(np.array([[year, genre_sentiment, holiday_release]]))[0]

# Example
predicted_rating = predict_rating(2024, 0.5, 1)  # Predict the rating for a movie in 2024 with holiday release and neutral genre sentiment
print(f'Predicted Rating for a movie in 2024 with holiday release: {predicted_rating:.2f}')

# Continue with your existing plots and CSV saving
# Save to CSV
df.to_csv('omdb_top_movies_with_sentiment_and_holiday.csv', index=False)

# Rotten Tomatoes data as Example
rt_data = {
    'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', '12 Angry Men', 'Schindler\'s List',
              'Pulp Fiction', 'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
              'Fight Club', 'Forrest Gump'],
    'RT_Rating': [91, 98, 94, 100, 97, 92, 95, 97, 79, 71]
}
rt_df = pd.DataFrame(rt_data)

# Merge IMDb and Rotten Tomatoes data on the movie title
combined_df = pd.merge(df, rt_df, on='Title', how='inner')

# Correlation between IMDb, Rotten Tomatoes, Genre Sentiment, and Holiday Release
correlation_matrix = combined_df[['Rating', 'RT_Rating', 'Genre_Sentiment', 'Is_Holiday_Release']].astype(float).corr()
print(correlation_matrix)

# Scatter plot to compare IMDb and Rotten Tomatoes ratings
plt.figure(figsize=(10,6))
plt.scatter(combined_df['Rating'].astype(float), combined_df['RT_Rating'].astype(float), alpha=0.5)
plt.xlabel('IMDb Rating')
plt.ylabel('Rotten Tomatoes Rating')
plt.title('Comparison of IMDb and Rotten Tomatoes Ratings')
plt.show()

# Scatter plot of IMDb rating vs Genre sentiment
plt.figure(figsize=(10,6))
plt.scatter(combined_df['Rating'].astype(float), combined_df['Genre_Sentiment'].astype(float), alpha=0.5, color='green')
plt.xlabel('IMDb Rating')
plt.ylabel('Genre Sentiment')
plt.title('IMDb Rating vs Genre Sentiment')
plt.show()

# Get the top 10 movies
top_10_movies = df.sort_values(by='Rating', ascending=False).head(10)
print(top_10_movies)

# Plot distribution of ratings
plt.hist(df['Rating'], bins=20, edgecolor='k')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of IMDb Ratings')
plt.show()

# Plot top 10 movies by rating
plt.figure(figsize=(10, 6))
plt.barh(top_10_movies['Title'], top_10_movies['Rating'], color='skyblue')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.title('Top 10 IMDb Movies by Rating')
plt.gca().invert_yaxis()  # To display the highest rating at the top
plt.show()