import requests
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime  # Import to get current year

# Your OMDb API key
api_key = '121c5367'

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get movie data from OMDb with error handling
def get_movie_data(title):
    params = {
        't': title,
        'apikey': api_key
    }
    try:
        response = requests.get('http://www.omdbapi.com/', params=params)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404 or 500)
        
        data = response.json()
        
        # Check if the API response was successful
        if data.get('Response') == 'True':
            return data
        else:
            print(f"Error: No data found for title '{title}' - Reason: {data.get('Error')}")
            return None  # Return None if the movie wasn't found or another issue occurred

    except requests.exceptions.RequestException as e:
        print(f"Request error for title '{title}': {e}")
        return None

# Function to analyze the sentiment of the movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']  # Use the compound score for overall sentiment

# Example list of top-rated movie titles to fetch
movie_titles = [
    'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
    '12 Angry Men', 'Schindler\'s List', 'Pulp Fiction',
    'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
    'Fight Club', 'Forrest Gump'
]

# Example budget data (in millions) for the movies
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

# Fetch data for each movie with error handling
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data:  # Only append if data was successfully retrieved
        movie_data.append(data)

# Check if we successfully fetched data before processing
if not movie_data:
    print("No movie data was retrieved. Exiting...")
    exit()

# Create DataFrame
df = pd.DataFrame(movie_data)

# Check if required columns are present
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 'imdbVotes']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns in data: {missing_columns}")
    exit()

# Select relevant columns and rename for clarity
df = df[required_columns]
df['Rating'] = df['imdbRating'].astype(float)

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Convert Year to a numeric feature and ensure Genre_Sentiment is float
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Calculate Director Popularity based on the number of movies directed in the dataset
df['Director_Popularity'] = df['Director'].map(df['Director'].value_counts())

# Convert Runtime from "X min" format to numeric (int)
df['Runtime'] = df['Runtime'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else np.nan)

# Add Budget data (in millions) to the DataFrame
df['Budget'] = df['Title'].map(budget_data)

# **New Feature: Director's Previous Success Rate**
# Calculate the average IMDb rating of movies directed by the same director
df['Directors_Avg_Rating'] = df.groupby('Director')['Rating'].transform('mean')

# **New Feature: Movie Popularity based on IMDb Votes**
df['Movie_Popularity'] = df['imdbVotes'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else 0)

# Features we will use to predict IMDb Rating
features = ['Year', 'Genre_Sentiment', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity', 'Directors_Avg_Rating']

# X = feature set (Year, Genre_Sentiment, Director_Popularity, Runtime, Budget, Movie_Popularity)
X = df[features]

# y = target variable (IMDb Rating)
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
print(f'R-squared: {r2}')  # R-squared tells us how well the model explains the variance in the data

# Display the actual vs predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Save to CSV
df.to_csv('omdb_top_movies_with_sentiment_and_director_popularity_and_popularity_and_director_success.csv', index=False)

# Example Rotten Tomatoes data
rt_data = {
    'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', '12 Angry Men', 'Schindler\'s List',
              'Pulp Fiction', 'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
              'Fight Club', 'Forrest Gump'],
    'RT_Rating': [91, 98, 94, 100, 97, 92, 95, 97, 79, 71]
}
rt_df = pd.DataFrame(rt_data)

# Merge IMDb and Rotten Tomatoes data on the movie title
combined_df = pd.merge(df, rt_df, on='Title', how='inner')

# Correlation between IMDb, Rotten Tomatoes, Genre Sentiment, and Budget
correlation_matrix = combined_df[['Rating', 'RT_Rating', 'Genre_Sentiment', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity', 'Directors_Avg_Rating']].astype(float).corr()
print(correlation_matrix)

# Scatter plot to compare IMDb and Rotten Tomatoes ratings
plt.figure(figsize=(10,6))
plt.scatter(combined_df['Rating'].astype(float), combined_df['RT_Rating'].astype(float), alpha=0.5)
plt.xlabel('IMDb Rating')
plt.ylabel('Rotten Tomatoes Rating')
plt.title('Comparison of IMDb and Rotten Tomatoes Ratings')
plt.show()

# Scatter plot of IMDb rating vs Director's Success Rate
plt.figure(figsize=(10,6))
plt.scatter(combined_df['Rating'].astype(float), combined_df['Directors_Avg_Rating'].astype(float), alpha=0.5, color='red')
plt.xlabel('IMDb Rating')
plt.ylabel('Director\'s Average Rating')
plt.title('IMDb Rating vs Director\'s Average Rating')
plt.show()
