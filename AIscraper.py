import requests
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Your OMDb API key
api_key = '121c5367'

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get movie data from OMDb 
def get_movie_data(title):
    params = {
        't': title,
        'apikey': api_key
    }
    response = requests.get('http://www.omdbapi.com/', params=params)
    return response.json()

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

# Fetch data for each movie
movie_data = []
for title in movie_titles:
    data = get_movie_data(title)
    if data['Response'] == 'True':
        movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director']]
df['Rating'] = df['imdbRating'].astype(float)

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Step 2: Prepare the data for prediction
# Convert Year to a numeric feature and ensure Genre_Sentiment is float
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Features we will use to predict IMDb Rating
features = ['Year', 'Genre_Sentiment']

# X = feature set (Year, Genre_Sentiment)
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

# Bonus: Make a prediction for a new movie
def predict_rating(year, genre_sentiment):
    return model.predict(np.array([[year, genre_sentiment]]))[0]

# Example usage
predicted_rating = predict_rating(2024, 0.5)  # Predict the rating for a movie in 2024 with neutral genre sentiment
print(f'Predicted Rating for a movie in 2024 with genre sentiment 0.5: {predicted_rating:.2f}')

# Continue with your existing plots and CSV saving
# Save to CSV
df.to_csv('omdb_top_movies_with_sentiment.csv', index=False)

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

# Correlation between IMDb, Rotten Tomatoes, and Genre Sentiment
correlation_matrix = combined_df[['Rating', 'RT_Rating', 'Genre_Sentiment']].astype(float).corr()
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
