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
import re


# Load OMDb API key from environment variable
api_key = os.getenv("OMDB_API_KEY")

if not api_key:
    print("Ooops: API key not found. Please set the OMDB_API_KEY environment variable.")
    exit(1)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Add production studio influence mapping
studio_influence = {
    "Warner Bros.": 10,
    "Paramount Pictures": 9,
    "Universal Pictures": 9,
    "20th Century Fox": 8,
    "Sony Pictures": 8,
    "New Line Cinema": 7,
    "Miramax Films": 7,
    "Lionsgate": 6,
    "DreamWorks Pictures": 8,
    "Columbia Pictures": 9,
}

director_popularity = {
    "Frank Darabont": 10,
    "Christopher Nolan": 9,
    "Quentin Tarantino": 8,
    "Steven Spielberg": 10,
    "Martin Scorsese": 9,
}

top_actors_list = ["Morgan Freeman", "Al Pacino", "Christian Bale", "Robert De Niro", "Leonardo DiCaprio"]

franchise_popularity = {
    "Marvel": 10,
    "Star Wars": 9,
    "Harry Potter": 8,
    "The Lord of the Rings": 9,
    "Fast & Furious": 7,
    "Batman": 9,
    "Spider-Man": 8,
    "James Bond": 8,
    "Transformers": 6,
    "The Avengers": 10,
}

# New feature: Extract Oscar nominations
def extract_oscar_nominations(awards):
    try:
        if isinstance(awards, str):
            match = re.search(r"(\d+)\s+nomination", awards, re.IGNORECASE)
            return int(match.group(1)) if match else 0
        return 0
    except Exception as e:
        print(f"Error extracting Oscar nominations: {e}")
        return 0

# Function to assign production studio influence score
def get_studio_influence(studio):
    """Assign an influence score to the production studio."""
    try:
        return studio_influence.get(studio, 5)  # Default to 5 for unknown studios
    except Exception as e:
        print(f"Error in get_studio_influence: {e}")
        return 5

# Function to categorize movies by runtime
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

# Fetch data for each movie
movie_titles = ["Inception", "The Dark Knight", "Titanic"]  # Replace with actual movie titles to fetch
movie_data = []
for title in movie_titles:
    try:
        data = get_movie_data(title)  # Replace with actual data fetching function
        if data:
            movie_data.append(data)
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")

# Create DataFrame
df = pd.DataFrame(movie_data)

# Apply new feature
df['Oscar_Nominations'] = df['Awards'].apply(extract_oscar_nominations)

# Error handling: Fill missing data or replace with defaults
try:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(datetime.now().year)  # Replace missing years with the current year
    df['Rating'] = pd.to_numeric(df['imdbRating'], errors='coerce').fillna(df['imdbRating'].median())  # Fill missing ratings with median
    df['Release_Date'] = pd.to_datetime(df['Released'], errors='coerce')  # Convert to datetime
    df['Movie_Age'] = df['Year'].apply(lambda x: datetime.now().year - x if pd.notnull(x) else 0)  # Handle missing years
except Exception as e:
    print(f"Error in data transformation: {e}")

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

# New Feature: International Release Indicator
def international_release_indicator(country):
    """
    Check if a movie was released in multiple countries.
    """
    if isinstance(country, str):
        countries = country.split(', ')
        return 1 if len(countries) > 1 else 0
    return 0

def box_office_success(budget, box_office):
    """
    Determine if a movie was a box office success based on its budget and earnings.
    Returns 1 if box office earnings are greater than the budget, else 0.
    """
    if budget > 0 and box_office > 0:
        return 1 if box_office > budget else 0
    return 0  # If budget or box office is missing, assume failure (0)

def genre_popularity(genre, df):
    """
    Calculate how popular the genre of the movie is based on the number of recent releases in that genre.
    The more recent releases in the genre, the higher the popularity.
    """
    if isinstance(genre, str):
        # Get the current year
        current_year = datetime.now().year
        # Filter the dataset for movies in the same genre
        genre_movies = df[df['Genre'].str.contains(genre, case=False, na=False)]
        # Count the number of movies in the genre released in the last 5 years
        recent_genre_movies = genre_movies[genre_movies['Year'] >= current_year - 5]
        return len(recent_genre_movies)
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

def director_success(director, df):
    """
    Calculate the average rating of movies directed by the same director.
    This represents the director's past success in terms of movie ratings.
    """
    if isinstance(director, str):
        # Filter the dataset for movies by this director
        director_movies = df[df['Director'] == director]
        # Calculate the average rating of these movies
        avg_rating = director_movies['Rating'].mean()
        return avg_rating if pd.notnull(avg_rating) else 0
    return 0

import requests

def fetch_social_media_mentions(title):
    """
    Fetch the number of social media mentions or hashtags for a movie.
    This can be integrated with APIs from platforms like Twitter, Instagram, or others.
    """
    # Replace with real API or method to fetch social media mentions
    api_url = f"https://api.socialmedia.com/search?query={title}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json().get('mention_count', 0)
    return 0

def audience_engagement_score(title):
    """
    Calculate the audience engagement score based on social media mentions.
    The higher the mentions, the higher the score.
    """
    mentions = fetch_social_media_mentions(title)
    # Normalize the mention count to a scale of 0-10
    return min(mentions / 100, 10)

# Apply the new feature to your DataFrame
df['Audience_Engagement_Score'] = df['Title'].apply(audience_engagement_score)

# Add it to the list of features for model training
features.append('Audience_Engagement_Score')

def director_previous_movie_rating_avg(director, df):
    """
    Calculate the average IMDb rating of a director's previous movies.
    This helps to capture the director's influence and consistency in movie quality.
    """
    if isinstance(director, str):
        # Filter the dataset for movies by this director
        director_movies = df[df['Director'] == director]
        
        # Calculate the average IMDb rating of previous movies
        avg_rating = director_movies['imdbRating'].mean()
        return avg_rating if not np.isnan(avg_rating) else 0.0
    return 0.0

# Apply the new feature to the DataFrame
df['Director_Previous_Movie_Rating_Avg'] = df['Director'].apply(lambda x: director_previous_movie_rating_avg(x, df))

# Add it to the list of features for model training
features.append('Director_Previous_Movie_Rating_Avg')

def director_previous_movie_rating_avg(director, df):
    """
    Calculate the average IMDb rating of a director’s past movies.
    """
    if isinstance(director, str):
        previous_movies = df[df['Director'] == director]['imdbRating'].dropna()
        if not previous_movies.empty:
            return previous_movies.mean()
    return 0  # Default if no previous movies found

df['Director_Previous_Movie_Rating_Avg'] = df['Director'].apply(lambda x: director_previous_movie_rating_avg(x, df))
features.append('Director_Previous_Movie_Rating_Avg')

def director_avg_rating(director, df):
    """
    Calculate the average IMDb rating of all movies directed by the given director.
    """
    if isinstance(director, str):
        director_movies = df[df['Director'] == director]
        if not director_movies.empty:
            return director_movies['imdbRating'].mean()
    return df['imdbRating'].mean()  # Default to dataset average if no data available

# Apply the function
df['Director_Avg_Rating'] = df['Director'].apply(lambda x: director_avg_rating(x, df))
features.append('Director_Avg_Rating')

def award_nomination_ratio(awards):
    """
    Calculate the ratio of awards won to total nominations.
    """
    try:
        if isinstance(awards, str):
            wins_match = re.search(r"(\d+)\s+win", awards, re.IGNORECASE)
            nominations_match = re.search(r"(\d+)\s+nomination", awards, re.IGNORECASE)
            
            wins = int(wins_match.group(1)) if wins_match else 0
            nominations = int(nominations_match.group(1)) if nominations_match else 1  # Avoid division by zero
            
            return wins / nominations
        return 0.0
    except Exception as e:
        print(f"Error calculating award-to-nomination ratio: {e}")
        return 0.0

# Apply the function
df['Award_Nomination_Ratio'] = df['Awards'].apply(award_nomination_ratio)
features.append('Award_Nomination_Ratio')

def box_office_success_score(budget, box_office):
    """
    Calculate a success score based on the ratio of box office earnings to budget.
    A higher ratio means better financial performance.
    """
    try:
        if budget > 0 and box_office > 0:
            return box_office / budget
        return 0.0
    except Exception as e:
        print(f"Error calculating box office success score: {e}")
        return 0.0

# Apply the function
df['Box_Office_Success_Score'] = df.apply(lambda row: box_office_success_score(row['Budget'], row['BoxOffice']), axis=1)
features.append('Box_Office_Success_Score')

def streaming_popularity_score(streaming_platforms):
    """
    Assign a popularity score based on the number of streaming platforms a movie is available on.
    """
    try:
        if isinstance(streaming_platforms, str):
            platforms = streaming_platforms.split(', ')
            return len(platforms)
        return 0
    except Exception as e:
        print(f"Error calculating streaming popularity score: {e}")
        return 0

# Example dataset column (Assuming 'Streaming' contains platform names)
df['Streaming_Popularity_Score'] = df['Streaming'].apply(streaming_popularity_score)
features.append('Streaming_Popularity_Score')

def award_winning_director_score(director):
    """
    Assign a score based on the number of major awards won by a director.
    """
    award_winning_directors = {
        "Steven Spielberg": 10,
        "Martin Scorsese": 9,
        "Christopher Nolan": 9,
        "Quentin Tarantino": 8,
        "James Cameron": 9,
        "Guillermo del Toro": 8,
        "Ridley Scott": 7,
        "Denis Villeneuve": 8,
        "Bong Joon-ho": 9
    }
    
    return award_winning_directors.get(director, 5)  # Default score for unknown directors

# Apply the function to the DataFrame
df['Director_Award_Score'] = df['Director'].apply(award_winning_director_score)
features.append('Director_Award_Score')

def audience_engagement_score(imdb_votes, num_reviews):
    """
    Calculate audience engagement score based on IMDb votes and number of reviews.
    """
    if pd.notnull(imdb_votes) and pd.notnull(num_reviews):
        return np.log1p(imdb_votes) + np.log1p(num_reviews)  # Apply log transformation to normalize
    return 0.0  # Default if data is missing

# Apply the function to the DataFrame
df['Audience_Engagement_Score'] = df.apply(lambda row: audience_engagement_score(row['imdbVotes'], row['num_reviews']), axis=1)
features.append('Audience_Engagement_Score')

def award_win_score(awards):
    """
    Assigns a weighted score based on the number of awards won.
    Oscars and Golden Globes have higher weight than other awards.
    """
    try:
        if isinstance(awards, str):
            oscars = re.search(r"Won (\d+) Oscar", awards, re.IGNORECASE)
            golden_globes = re.search(r"Won (\d+) Golden Globe", awards, re.IGNORECASE)
            total_wins = re.search(r"Won (\d+) award", awards, re.IGNORECASE)

            oscar_wins = int(oscars.group(1)) * 5 if oscars else 0
            golden_globe_wins = int(golden_globes.group(1)) * 3 if golden_globes else 0
            other_wins = int(total_wins.group(1)) if total_wins else 0

            return oscar_wins + golden_globe_wins + other_wins
        return 0
    except Exception as e:
        print(f"Error calculating award win score: {e}")
        return 0

# Apply the function to the DataFrame
df['Award_Win_Score'] = df['Awards'].apply(award_win_score)
features.append('Award_Win_Score')


def audience_engagement_score(imdb_votes, imdb_rating):
    """
    Calculate an audience engagement score based on IMDb votes and rating.
    A higher score suggests a more engaged audience.
    """
    try:
        if imdb_votes > 0 and imdb_rating > 0:
            return imdb_votes * imdb_rating / 100000  # Normalize score
        return 0.0
    except Exception as e:
        print(f"Error calculating audience engagement score: {e}")
        return 0.0

# Apply the function
df['Audience_Engagement_Score'] = df.apply(lambda row: audience_engagement_score(row['imdbVotes'], row['imdbRating']), axis=1)
features.append('Audience_Engagement_Score')

def audience_engagement_score(imdb_votes, imdb_rating):
    """
    Calculate an audience engagement score based on IMDb votes and rating.
    A higher score suggests a more engaged audience.
    """
    try:
        if imdb_votes > 0 and imdb_rating > 0:
            return imdb_votes * imdb_rating / 100000  # Normalize score
        return 0.0
    except Exception as e:
        print(f"Error calculating audience engagement score: {e}")
        return 0.0

def add_release_season(df, features):
    """
    Add a new feature for the release season of the movie.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of feature column names.

    Returns:
        pd.DataFrame, list: Updated DataFrame and features list.
    """
    def classify_season(month):
        """Classify the movie's release month into a season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Release_Season'] = df['Release_Month'].apply(classify_season)
    df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)
    features += [col for col in df.columns if col.startswith('Release_Season_')]
    return df, features

def box_office_success(budget, box_office):
    """
    Calculate the success of a movie based on its budget and box office revenue.
    """
    try:
        if budget > 0:
            return box_office / budget
        return 0.0
    except Exception as e:
        print(f"Error calculating box office success: {e}")
        return 0.0

def genre_popularity(genre, df):
    """
    Calculate the popularity of a genre based on its occurrence in the dataset.
    """
    try:
        return df[df['Genre'] == genre].shape[0] / df.shape[0]
    except Exception as e:
        print(f"Error calculating genre popularity: {e}")
        return 0.0

def director_success(director, df):
    """
    Calculate the success of a director based on the box office performance of their movies.
    """
    try:
        director_movies = df[df['Director'] == director]
        if len(director_movies) > 0:
            return director_movies['BoxOffice'].mean()
        return 0.0
    except Exception as e:
        print(f"Error calculating director success: {e}")
        return 0.0


def add_release_season(df, features):
    """
    Add a new feature for the release season of the movie.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of feature column names.

    Returns:
        pd.DataFrame, list: Updated DataFrame and features list.
    """
    def classify_season(month):
        """Classify the movie's release month into a season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['Release_Season'] = df['Release_Month'].apply(classify_season)
    df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)
    features += [col for col in df.columns if col.startswith('Release_Season_')]
    return df, features



def cast_diversity_score(cast):
    """
    Calculate the diversity score based on the number of unique nationalities in the cast.
    """
    try:
        if isinstance(cast, str):
            # Assuming the cast information includes the names and nationalities of the actors
            cast_nationalities = cast.split(', ')
            unique_nationalities = set([name.split('(')[-1].strip(')') for name in cast_nationalities])
            return len(unique_nationalities)
        return 0
    except Exception as e:
        print(f"Error calculating cast diversity score: {e}")
        return 0



def social_media_buzz_score(social_media_mentions, social_media_likes):
    """
    Calculate a social media buzz score based on mentions and likes.
    A higher score suggests higher popularity and engagement.
    """
    try:
        if social_media_mentions > 0 and social_media_likes > 0:
            return (social_media_mentions + social_media_likes) / 1000  # Normalize score
        return 0.0
    except Exception as e:
        print(f"Error calculating social media buzz score: {e}")
        return 0.0


def sentiment_score(review):
    """
    Calculate a sentiment score for a given review text.
    A higher score suggests a more positive sentiment.
    """
    try:
        if isinstance(review, str):
            analysis = TextBlob(review)
            return analysis.sentiment.polarity  # Sentiment polarity score (-1 to 1)
        return 0.0
    except Exception as e:
        print(f"Error calculating sentiment score: {e}")
        return 0.0

import pandas as pd

def genre_complexity_score(genres):
    """
    Calculate a genre complexity score based on the number of genres a movie belongs to.
    A higher score suggests a broader appeal across different audience segments.
    """
    try:
        if isinstance(genres, str):
            genre_list = genres.split(', ')
            return len(genre_list)
        return 0
    except Exception as e:
        print(f"Error calculating genre complexity score: {e}")
        return 0

import pandas as pd

def film_festival_participation_score(festivals):
    """
    Calculate a score based on the number of film festivals a movie has participated in.
    A higher score suggests greater industry recognition and prestige.
    """
    try:
        if isinstance(festivals, str):
            festival_list = festivals.split(', ')
            return len(festival_list)
        return 0
    except Exception as e:
        print(f"Error calculating film festival participation score: {e}")
        return 0

def award_wins_score(awards):
    """
    Calculate a score based on the number of awards a movie has won.
    A higher score suggests greater recognition and acclaim.
    """
    try:
        if isinstance(awards, str):
            awards_list = awards.split(', ')
            return len(awards_list)
        return 0
    except Exception as e:
        print(f"Error calculating award wins score: {e}")
        return 0

def budget_to_box_office_ratio(budget, box_office):
    """
    Calculate the ratio of the box office revenue to the production budget of a movie.
    A higher ratio indicates a more profitable movie.
    """
    try:
        if budget > 0:
            return box_office / budget
        return 0.0
    except Exception as e:
        print(f"Error calculating budget to box office ratio: {e}")
        return 0.0

def critic_review_score(critic_reviews):
    """
    Calculate the average score from critic reviews.
    A higher score suggests better critical reception.
    """
    try:
        if isinstance(critic_reviews, list) and len(critic_reviews) > 0:
            return sum(critic_reviews) / len(critic_reviews)
        return 0.0
    except Exception as e:
        print(f"Error calculating critic review score: {e}")
        return 0.0

def audience_review_score(audience_reviews):
    """
    Calculate the average score from audience reviews.
    A higher score suggests better audience reception.
    """
    try:
        if isinstance(audience_reviews, list) and len(audience_reviews) > 0:
            return sum(audience_reviews) / len(audience_reviews)
        return 0.0
    except Exception as e:
        print(f"Error calculating audience review score: {e}")
        return 0.0

def sequel_potential_score(box_office, audience_score, critic_score):
    """
    Calculate a score that estimates the potential success of a sequel.
    Factors considered include box office performance, audience reception, and critical reviews.
    """
    try:
        if box_office > 0 and audience_score > 0 and critic_score > 0:
            return (box_office / 1000000) * 0.5 + audience_score * 0.3 + critic_score * 0.2
        return 0.0
    except Exception as e:
        print(f"Error calculating sequel potential score: {e}")
        return 0.0

def calculate_marketing_spend(movie_budget):
    """
    Estimate marketing spend based on the movie's budget.
    """
    return movie_budget * 0.3  # Assuming 30% of the budget is spent on marketing

def calculate_average_rating(imdb_rating, rt_rating, metacritic_rating):
    """
    Calculate the average rating based on ratings from IMDb, Rotten Tomatoes, and Metacritic.
    """
    return (imdb_rating + rt_rating + metacritic_rating) / 3

def count_genres(genres):
    """
    Count the number of unique genres a movie belongs to.
    """
    return len(set(genres.split(',')))

def calculate_director_success(director_name, movie_data):
    """
    Calculate a score representing the director's past success based on previous box office performance.
    """
    director_movies = movie_data[movie_data['Director'] == director_name]
    return director_movies['BoxOffice'].sum() / len(director_movies) if len(director_movies) > 0 else 0

# Add the new feature to the dataset
df['Director_Success'] = df['Director'].apply(lambda director: calculate_director_success(director, df))

# Add the new feature to the feature list
features.append('Director_Success')

# Re-train the model with the updated features
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Recalculate predictions and metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Updated Mean Squared Error: {mse}')
print(f'Updated R-squared: {r2}')
