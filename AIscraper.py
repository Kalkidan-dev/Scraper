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
API_KEY = '121c5367'

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get movie data from OMDb with error handling
def get_movie_data(title):
    params = {'t': title, 'apikey': API_KEY}
    try:
        response = requests.get('http://www.omdbapi.com/', params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('Response') == 'True':
            return data
        print(f"Error: No data found for '{title}' - Reason: {data.get('Error')}")
    except requests.exceptions.RequestException as e:
        print(f"Request error for '{title}': {e}")
    return None

# Data processing functions
def analyze_genre_sentiment(genre):
    return analyzer.polarity_scores(genre)['compound'] if isinstance(genre, str) else 0

def convert_box_office_to_numeric(box_office):
    return int(box_office[1:].replace(',', '')) if isinstance(box_office, str) and box_office.startswith('$') else 0

def extract_awards_count(awards):
    return sum(map(int, re.findall(r'\d+', awards))) if isinstance(awards, str) else 0

def calculate_genre_diversity(genre):
    genres = genre.split(',') if isinstance(genre, str) else []
    return len(set(genres)) / len(genres) if genres else 0

def release_month_sentiment(release_date):
    try:
        release_month = datetime.strptime(release_date, '%d %b %Y').month
        month_sentiment = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.8, 6: 0.9, 
                           7: 1.0, 8: 0.9, 9: 0.4, 10: 0.6, 11: 0.7, 12: 0.5}
        return month_sentiment.get(release_month, 0.0)
    except (ValueError, TypeError):
        return 0.0

def calculate_director_metrics(director, ratings_df):
    director_movies = ratings_df[ratings_df['Director'] == director]
    avg_rating = director_movies['Rating'].mean() if not director_movies.empty else 0
    movie_count = director_movies.shape[0]
    return avg_rating, movie_count

def calculate_cast_popularity(cast, ratings_df):
    if isinstance(cast, str):
        actors = cast.split(', ')
        total_popularity = sum(ratings_df[ratings_df['Actors'].str.contains(actor, na=False)]['Movie_Popularity'].sum() 
                               for actor in actors)
        return total_popularity / len(actors) if actors else 0
    return 0

# Movie titles to fetch
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

# Fetch movie data
movie_data = [data for title in movie_titles if (data := get_movie_data(title))]

if not movie_data:
    print("No movie data retrieved. Exiting...")
    exit()

# Create DataFrame and process data
df = pd.DataFrame(movie_data)
df['Rating'] = df['imdbRating'].astype(float)
df['Year'] = df['Year'].astype(int)
df['Runtime'] = df['Runtime'].apply(lambda x: int(x.split()[0]) if isinstance(x, str) else np.nan)
df['Budget'] = df['Title'].map(budget_data)
df['Movie_Popularity'] = df['imdbVotes'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else 0)
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)
df['Awards_Count'] = df['Awards'].apply(extract_awards_count)
df['Num_Genres'] = df['Genre'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
df['Genre_Diversity'] = df['Genre'].apply(calculate_genre_diversity)
df['Release_Month_Sentiment'] = df['Released'].apply(release_month_sentiment)
df['Director_Movie_Count'] = df['Director'].apply(lambda x: calculate_director_metrics(x, df)[1])
df['Cast_Popularity'] = df['Actors'].apply(lambda x: calculate_cast_popularity(x, df))
df['BoxOffice'] = df['BoxOffice'].apply(convert_box_office_to_numeric)
df['BoxOffice_per_Genre'] = df.apply(
    lambda row: row['BoxOffice'] / row['Num_Genres'] if row['Num_Genres'] > 0 else 0, axis=1
)
df['Rating_per_Genre'] = df.apply(
    lambda row: row['Rating'] / row['Num_Genres'] if row['Num_Genres'] > 0 else 0, axis=1
)
df['Movie_Age'] = datetime.now().year - df['Year']

# Define features and target
features = ['Year', 'Genre_Sentiment', 'Runtime', 'Budget', 'Movie_Popularity', 
            'Awards_Count', 'Genre_Diversity', 'Release_Month_Sentiment', 
            'Director_Movie_Count', 'Cast_Popularity', 'BoxOffice_per_Genre', 
            'Num_Genres', 'Rating_per_Genre', 'Movie_Age']
X = df[features]
y = df['Rating']

# Train-test split and modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(df['Cast_Popularity'], df['Rating'], alpha=0.5, color='green')
plt.xlabel('Cast Popularity')
plt.ylabel('IMDb Rating')
plt.title('IMDb Rating vs Cast Popularity')
plt.show()
