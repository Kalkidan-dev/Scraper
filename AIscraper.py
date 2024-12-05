import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime
import re
import requests
from sklearn.impute import SimpleImputer

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
    return sentiment_score['compound']

# Function to convert BoxOffice to numeric
def convert_box_office_to_numeric(box_office):
    if isinstance(box_office, str) and box_office.startswith('$'):
        return int(box_office[1:].replace(',', ''))
    return 0

# Extracting Awards Count
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

# New Feature: Sequel Indicator
def is_sequel(title):
    sequels = ['2', 'II', 'III', 'IV', 'V']
    for sequel in sequels:
        if sequel in title:
            return 1
    return 0

# New Feature: Lead Actor Popularity
def actor_popularity(actors):
    if isinstance(actors, str):
        actor_list = actors.split(', ')
        actor_popularity = 0
        for actor in actor_list:
            actor_data = get_movie_data(actor)  # Fetch data for each actor's filmography
            if actor_data:
                actor_popularity += actor_data.get('imdbVotes', 0)  # Add IMDb votes of the actor's movies
        return actor_popularity
    return 0

# New Feature: Director's Previous Success
def director_previous_success(director, df):
    if isinstance(director, str):
        director_movies = df[df['Director'] == director]
        return director_movies['BoxOffice'].sum()  # Sum of BoxOffice earnings of all the movies directed by the same director
    return 0

# New Feature: Movie Popularity Trend
def movie_popularity_trend(row):
    if row['BoxOffice'] > 0 and row['Rating'] > 7.0:
        # If the movie has a good box office and a high IMDb rating, we assume a positive trend
        return 1
    elif row['BoxOffice'] < 100000000 and row['Rating'] < 6.0:
        # If the movie has a low box office and a low IMDb rating, we assume a negative trend
        return 0
    else:
        return 1 if row['Rating'] > 6.0 else 0

# New Feature: Director's Reputation
def director_reputation(director, df):
    if isinstance(director, str):
        director_movies = df[df['Director'] == director]
        return director_movies['Rating'].mean()  # Average IMDb rating of the director's movies
    return 0

# New Feature: Production Company Popularity
def production_company_popularity(production_companies, df):
    if isinstance(production_companies, str):
        company_list = production_companies.split(', ')
        company_popularity = 0
        for company in company_list:
            company_movies = df[df['Production'] == company]
            company_popularity += company_movies['Rating'].mean()  # Average IMDb rating of the company's movies
        return company_popularity
    return 0

# New Feature: Director's Age
def director_age(director, df):
    if isinstance(director, str):
        director_data = get_movie_data(director)  # Fetch data for director's birth year
        if director_data and director_data.get('Director') == director:
            birth_year = director_data.get('Born')
            if birth_year:
                try:
                    birth_year = int(birth_year.split()[-1])  # Extract year from the birth string
                    return datetime.now().year - birth_year
                except ValueError:
                    return 0
    return 0

# New Feature: Movie Release Period (Season)
def release_period(release_date):
    if isinstance(release_date, str) and release_date:
        try:
            release_month = datetime.strptime(release_date, '%d %b %Y').month
            if release_month in [6, 7, 8]:
                return 'Summer'
            elif release_month in [9, 10, 11]:
                return 'Fall'
            elif release_month in [12, 1, 2]:
                return 'Winter'
            elif release_month in [3, 4, 5]:
                return 'Spring'
        except ValueError:
            return 'Unknown'
    return 'Unknown'

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
required_columns = ['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Runtime', 'imdbVotes', 'BoxOffice', 'Awards', 'Released', 'Actors', 'Production']
df = df[required_columns]
df['Rating'] = df['imdbRating'].astype(float)

# Add new features to the DataFrame
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)
df['BoxOffice'] = df['BoxOffice'].apply(convert_box_office_to_numeric)
df['Awards_Count'] = df['Awards'].apply(extract_awards_count)
df['Genre_Diversity'] = df['Genre'].apply(calculate_genre_diversity)
df['Release_Month_Sentiment'] = df['Released'].apply(release_month_sentiment)
df['Actor_Diversity'] = df['Actors'].apply(calculate_actor_diversity)
df['Weekend_Release'] = df['Released'].apply(is_weekend_release)
df['Sequel_Indicator'] = df['Title'].apply(is_sequel)
df['Actor_Popularity'] = df['Actors'].apply(actor_popularity)
df['Director_Previous_Success'] = df['Director'].apply(lambda x: director_previous_success(x, df))
df['Movie_Popularity_Trend'] = df.apply(movie_popularity_trend, axis=1)
df['Director_Reputation'] = df['Director'].apply(lambda x: director_reputation(x, df))
df['Production_Company_Popularity'] = df['Production'].apply(lambda x: production_company_popularity(x, df))
df['Director_Age'] = df['Director'].apply(lambda x: director_age(x, df))
df['Release_Period'] = df['Released'].apply(release_period)

# Define feature set
features = [
    'Genre_Sentiment', 'Year', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity',
    'Num_Genres', 'Rating_per_Genre', 'Movie_Age', 'Weekend_Release', 'Sequel_Indicator',
    'Actor_Diversity', 'Production_Company_Popularity', 'Director_Age', 'Release_Period'
]

X = df[features]
y = df['Rating']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

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

# Visualize the impact of Director Age on Ratings
plt.figure(figsize=(10, 6))
plt.scatter(df['Director_Age'], df['Rating'], alpha=0.5, color='blue')
plt.xlabel('Director Age')
plt.ylabel('IMDb Rating')
plt.title('IMDb Rating vs Director Age')
plt.show()
