import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import requests
import re

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Your OMDb API key
api_key = '121c5367'

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
        
# New Feature: International Release Indicator
def international_release_indicator(country):
    """
    Check if a movie was released in multiple countries.
    """
    if isinstance(country, str):
        countries = country.split(', ')
        return 1 if len(countries) > 1 else 0
    return 0
# New Feature: Critics vs Audience Rating Disparity
def critics_vs_audience_disparity(critics_rating, audience_rating):
    """
    Calculate the absolute difference between critics' and audience ratings.
    """
    if critics_rating is not None and audience_rating is not None:
        return abs(critics_rating - audience_rating)
    return 0.0

# New Feature: Budget-to-BoxOffice Ratio
def budget_to_boxoffice_ratio(budget, box_office):
    """
    Calculate the ratio of budget to box office earnings.
    """
    if budget > 0 and box_office > 0:
        return budget / box_office
    return 0.0
# New Feature: Franchise Indicator
def franchise_indicator(title):
    """
    Check if a movie title contains keywords commonly associated with franchises.
    """
    franchise_keywords = ['Marvel', 'Avengers', 'Star Wars', 'Harry Potter', 
                          'Fast & Furious', 'Mission Impossible', 'Transformers', 
                          'Spider-Man', 'Batman', 'Superman', 'Jurassic']
    for keyword in franchise_keywords:
        if keyword.lower() in title.lower():
            return 1
    return 0

# New Feature: Holiday Release Indicator
def holiday_release_indicator(release_date):
    """
    Check if the release date falls within a holiday season.
    """
    holidays = {
        'New Year': ['01-01'],
        'Valentine\'s Day': ['02-14'],
        'Thanksgiving': ['11-24', '11-25', '11-26', '11-27', '11-28'],
        'Christmas': ['12-24', '12-25', '12-26']
    }
    
    if isinstance(release_date, str) and release_date:
        try:
            release_month_day = datetime.strptime(release_date, '%d %b %Y').strftime('%m-%d')
            for holiday, dates in holidays.items():
                if release_month_day in dates:
                    return 1
        except ValueError:
            return 0
    return 0
# New Feature: Seasonal Genre Popularity
def seasonal_genre_popularity(genre, release_date):
    """
    Assign a score to the genre based on its popularity during the release month.
    """
    seasonal_popularity = {
        'January': {'Drama': 0.7, 'Action': 0.4, 'Comedy': 0.6},
        'February': {'Romance': 1.0, 'Drama': 0.8, 'Comedy': 0.7},
        'March': {'Action': 0.9, 'Thriller': 0.8, 'Comedy': 0.5},
        'April': {'Comedy': 0.8, 'Family': 0.9, 'Drama': 0.6},
        'May': {'Action': 1.0, 'Adventure': 0.9, 'Sci-Fi': 0.8},
        'June': {'Action': 0.9, 'Adventure': 1.0, 'Family': 0.9},
        'July': {'Action': 1.0, 'Adventure': 1.0, 'Comedy': 0.8},
        'August': {'Action': 0.8, 'Thriller': 0.7, 'Comedy': 0.6},
        'September': {'Drama': 0.9, 'Thriller': 0.8, 'Action': 0.6},
        'October': {'Horror': 1.0, 'Thriller': 0.9, 'Action': 0.5},
        'November': {'Drama': 0.8, 'Family': 1.0, 'Romance': 0.7},
        'December': {'Family': 1.0, 'Drama': 0.9, 'Fantasy': 0.8},
    }
    
    if isinstance(genre, str) and isinstance(release_date, str) and release_date:
        try:
            release_month = datetime.strptime(release_date, '%d %b %Y').strftime('%B')
            genres = genre.split(', ')
            total_score = 0
            for g in genres:
                total_score += seasonal_popularity.get(release_month, {}).get(g.strip(), 0)
            return total_score / len(genres) if genres else 0
        except ValueError:
            return 0
    return 0

# New Feature: Star Power Index
def star_power_index(actors, director):
    """
    Calculate the star power index based on actors' and director's popularity.
    """
    # Predefined popularity scores (replace or expand this dictionary with real data if available)
    predefined_popularity = {
        'Robert Downey Jr.': 95, 'Scarlett Johansson': 90, 'Leonardo DiCaprio': 92,
        'Chris Evans': 88, 'Brad Pitt': 85, 'Angelina Jolie': 87, 'Tom Cruise': 93,
        'Christopher Nolan': 97, 'Steven Spielberg': 96, 'Quentin Tarantino': 94,
        'Martin Scorsese': 95, 'James Cameron': 98, 'Ridley Scott': 92
    }
    
    # Calculate actors' popularity
    actor_score = 0
    if isinstance(actors, str):
        actor_list = actors.split(', ')
        actor_score = sum(predefined_popularity.get(actor, 50) for actor in actor_list) / len(actor_list)
    
    # Calculate director's popularity
    director_score = predefined_popularity.get(director, 50) if isinstance(director, str) else 50
    
    # Combine scores into a Star Power Index
    return (actor_score + director_score) / 2

# New Feature: Genre Familiarity Index
def genre_familiarity_index(director, genre, df):
    """
    Calculate how familiar a director is with the genres of the current movie.
    """
    if isinstance(director, str) and isinstance(genre, str):
        # Split the genres into a list
        genres = genre.split(', ')
        # Filter the dataset for movies by this director
        director_movies = df[df['Director'] == director]
        # Count the occurrence of each genre in the director's previous movies
        genre_count = sum(director_movies['Genre'].str.contains(g, case=False, na=False).sum() for g in genres)
        # Normalize by the number of genres in the current movie
        return genre_count / len(genres) if len(genres) > 0 else 0
    return 0
# New Feature: Seasonal Popularity Score
def seasonal_popularity_score(release_date):
    """
    Assign a seasonal popularity score based on the release date.
    """
    if isinstance(release_date, str) and release_date:
        try:
            # Parse the release date to extract the month
            release_month = datetime.strptime(release_date, '%d %b %Y').month
            
            # Seasonal popularity scores (values can be adjusted based on real data)
            season_scores = {
                'Winter': 0.8,  # December, January, February
                'Spring': 0.7,  # March, April, May
                'Summer': 1.0,  # June, July, August
                'Fall': 0.6     # September, October, November
            }
            
            # Assign seasons based on the month
            if release_month in [12, 1, 2]:
                return season_scores['Winter']
            elif release_month in [3, 4, 5]:
                return season_scores['Spring']
            elif release_month in [6, 7, 8]:
                return season_scores['Summer']
            elif release_month in [9, 10, 11]:
                return season_scores['Fall']
        except ValueError:
            return 0.0
    return 0.0
# New Feature: Director Collaboration Frequency
def director_collaboration_frequency(director, actors, df):
    """
    Calculate how often a director collaborates with the same actors.
    """
    if isinstance(director, str) and isinstance(actors, str):
        # Split the actors into a list
        actor_list = actors.split(', ')
        
        # Filter the dataset for movies by the same director
        director_movies = df[df['Director'] == director]
        
        # Count collaborations with the same actors
        collaboration_count = 0
        for actor in actor_list:
            collaboration_count += director_movies['Actors'].apply(lambda x: actor in x if isinstance(x, str) else False).sum()
        
        return collaboration_count / len(actor_list) if len(actor_list) > 0 else 0
    return 0
# New feature: Categorize movies by runtime
def categorize_movie_length(runtime):
    """
    Categorize movies into 'Short', 'Average', and 'Long' based on runtime.
    """
    if pd.notnull(runtime):
        if runtime < 90:
            return "Short"
        elif 90 <= runtime <= 150:
            return "Average"
        else:
            return "Long"
    return "Unknown"  # Handle missing or unknown runtime
 # New feature: Calculate critical reception sentiment
def get_critical_reception_sentiment(plot):
    """
    Use VADER sentiment analysis to calculate the sentiment of the plot.
    """
    if isinstance(plot, str):
        sentiment = analyzer.polarity_scores(plot)
        return sentiment['compound']
    return 0  # Default to neutral sentiment if plot is missing
 

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

# Function to simulate social media buzz
def generate_social_media_buzz(title):
    """
    Simulate social media buzz as a random number.
    Replace this with real data from a social media API if available.
    """
    return random.randint(1000, 1000000)  # Simulate mentions between 1k and 1M

# New Feature: Count Main Actors' Popularity Score
def actors_popularity_score(actors):
    """
    Simulate actors' popularity based on predefined scores.
    Replace with real data if available.
    """
    predefined_popularity = {
        'Robert Downey Jr.': 95, 'Scarlett Johansson': 90, 'Leonardo DiCaprio': 92,
        'Chris Evans': 88, 'Brad Pitt': 85, 'Angelina Jolie': 87, 'Tom Cruise': 93
    }
    if isinstance(actors, str):
        actors_list = actors.split(', ')
        total_score = sum(predefined_popularity.get(actor, 50) for actor in actors_list)
        return total_score / len(actors_list)
    return 50

# New Feature: Director Experience
def director_experience(director, df):
    """
    Calculate the number of movies a director has directed before.
    """
    if isinstance(director, str):
        return len(df[df['Director'] == director])
    return 0

# New Feature: Audience Sentiment Indicator
def audience_sentiment_indicator(audience_reviews):
    """
    Analyze the sentiment of the audience reviews.
    A positive score is positive sentiment, 0 is neutral, and negative is negative sentiment.
    """
    if isinstance(audience_reviews, str):
        sentiment_score = analyzer.polarity_scores(audience_reviews)
        return sentiment_score['compound']
    return 0.0  # Default neutral sentiment if no audience reviews available

# Add all features from previous and new ones
features = [
    'Year', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity',
    'Genre_Sentiment', 'BoxOffice', 'Awards_Count', 'Genre_Diversity',
    'Release_Month_Sentiment', 'Weekend_Release', 'Sequel', 
    'Critic_Reviews_Sentiment', 'Audience_Engagement_Score', 'Social_Media_Buzz',
    'Actors_Popularity_Score', 'Director_Experience', 'Budget_to_BoxOffice_Ratio',
    'Average_Review_Score', 'Director_Genre_Specialization', 'Critical_Acclaim_Indicator',
    'Audience_Sentiment_Indicator'  # New feature added
]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

# Define the target variable (e.g., 'Rating')
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RandomForestRegressor and GridSearchCV parameters
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearch
best_rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)

# Print evaluation metrics
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Visualize Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Regressor - Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
