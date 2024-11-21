import requests
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime

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
    if data:
        movie_data.append(data)

if not movie_data:
    print("No movie data was retrieved. Exiting...")
    exit()

# Create DataFrame
df = pd.DataFrame(movie_data)

# Check for required columns
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 'imdbVotes', 'BoxOffice']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns in data: {missing_columns}")
    exit()

# Select relevant columns
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

# **New Feature: Box Office Revenue per Genre**
def convert_box_office_to_numeric(box_office):
    if isinstance(box_office, str) and box_office.startswith('$'):
        return int(box_office[1:].replace(',', ''))
    return 0

df['BoxOffice'] = df['BoxOffice'].apply(convert_box_office_to_numeric)
df['BoxOffice_per_Genre'] = df.apply(
    lambda row: row['BoxOffice'] / row['Num_Genres'] if row['Num_Genres'] > 0 else 0, axis=1
)

# Features for the model
features = ['Year', 'Genre_Sentiment', 'Director_Popularity', 'Runtime', 
            'Budget', 'Movie_Popularity', 'Num_Genres', 'Rating_per_Genre', 'Movie_Age', 'BoxOffice_per_Genre']

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

# Display actual vs predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Save to CSV
df.to_csv('omdb_with_boxoffice_per_genre.csv', index=False)

# Scatter plot: IMDb Rating vs Box Office Revenue per Genre
plt.figure(figsize=(10,6))
plt.scatter(df['Rating'], df['BoxOffice_per_Genre'], alpha=0.5, color='blue')
plt.xlabel('IMDb Rating')
plt.ylabel('Box Office Revenue per Genre')
plt.title('IMDb Rating vs Box Office Revenue per Genre')
plt.show()
