import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

# Assuming movie_data is a list where each item is a dictionary with movie details
movie_data = []

# Step 1: Data Retrieval and Cleaning (Assume 'data' is being appended successfully)
if data:  # Only append if data was successfully retrieved
    movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Release_Date', 'Awards', 'Runtime', 'Actors']]
df['Rating'] = df['imdbRating'].astype(float)

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Step 2: Prepare Release Date Features
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Month'] = df['Release_Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['Season'] = df['Month'].apply(get_season)
df = pd.get_dummies(df, columns=['Season'], drop_first=True)
df['Day_of_Week'] = df['Release_Date'].dt.day_name()
df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

def is_holiday_release(date):
    holiday_dates = ['12-25', '11-26']
    if pd.isna(date): return 0
    return int(date.strftime('%m-%d') in holiday_dates)

df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release)
df['Is_Peak_Season'] = df['Month'].isin([6, 7, 8]).astype(int)

# Step 3: Prepare the data for prediction
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# New Feature: Box Office Success Score
def calculate_box_office_success(budget, revenue):
    if revenue > 0:
        return (revenue / max(budget, 1)) + (revenue / 1e6)  # Normalize with a scaling factor
    return 0

df['Box_Office_Success_Score'] = df.apply(lambda x: calculate_box_office_success(x['Budget'], x['Revenue']), axis=1)

# New Feature: Genre Diversity Score
def calculate_genre_diversity(genres):
    if pd.isna(genres):
        return 0
    return len(set(genres.split(',')))  # Count unique genres

df['Genre_Diversity_Score'] = df['Genre'].apply(calculate_genre_diversity)

# New Feature: Actor's Award-Winning History
def has_actor_won_award(actor):
    award_winners = {
        'Leonardo DiCaprio': 1,  # Won an Oscar
        'Robert Downey Jr.': 1,  # Multiple awards
        'Meryl Streep': 1,  # Multiple Oscars
        'Chris Hemsworth': 0,  # No Oscar yet
    }
    return award_winners.get(actor, 0)  # Default to 0 if not found

df['Lead_Actor_Has_Award'] = df['Lead_Actor'].apply(has_actor_won_award)

# New Feature: Sequel Indicator
def is_sequel(title):
    sequel_keywords = ['2', 'II', '3', 'III', '4', 'IV', 'Returns', 'Revenge', 'Part', 'Chapter']
    return int(any(keyword in title for keyword in sequel_keywords))

df['Is_Sequel'] = df['Title'].apply(is_sequel)

# New Feature: Critical Acclaim Score
def calculate_critical_acclaim(imdb, rt, metacritic):
    scores = []
    if not pd.isna(imdb): scores.append(imdb * 10)  # Scale IMDb to 100
    if not pd.isna(rt): scores.append(rt)  # RT is already on a 100 scale
    if not pd.isna(metacritic): scores.append(metacritic)  # Metacritic is on 100 scale
    return np.mean(scores) if scores else np.nan

df['Critical_Acclaim_Score'] = df.apply(lambda x: calculate_critical_acclaim(
    x.get('imdbRating'), x.get('Rotten_Tomatoes_Score'), x.get('Metacritic_Score')), axis=1)

# New Feature: Movie Hype Score
def calculate_hype_score(user_reviews, trailer_views, social_media_mentions):
    # Normalize values to avoid large scale differences
    reviews_weight = user_reviews / 1000 if user_reviews else 0
    trailer_weight = trailer_views / 100000 if trailer_views else 0
    social_weight = social_media_mentions / 10000 if social_media_mentions else 0
    
    return (reviews_weight * 0.5) + (trailer_weight * 0.3) + (social_weight * 0.2)

df['Movie_Hype_Score'] = df.apply(lambda x: calculate_hype_score(
    x.get('User_Reviews_Count'), x.get('Trailer_Views'), x.get('Social_Media_Mentions')), axis=1)

# New Feature: Production Budget Category
def categorize_budget(budget):
    if budget < 10_000_000: 
        return 'Low Budget'
    elif budget < 100_000_000: 
        return 'Mid Budget'
    else: 
        return 'High Budget'

df['Budget_Category'] = df['Budget'].apply(lambda x: categorize_budget(x) if not pd.isna(x) else 'Unknown')

# Convert to dummy variables for model training
df = pd.get_dummies(df, columns=['Budget_Category'], drop_first=True)

# New Feature: Studio Reputation Score
def calculate_studio_reputation(studio):
    past_movies = df[df['Studio'] == studio]
    if not past_movies.empty:
        return (past_movies['Rating'].mean() * 0.7) + (past_movies['Revenue'].mean() / 1_000_000 * 0.3)
    return 5.0  # Default reputation score for unknown studios

df['Studio_Reputation_Score'] = df['Studio'].apply(lambda x: calculate_studio_reputation(x) if pd.notna(x) else 5.0)

from textblob import TextBlob

# New Feature: Trailer Sentiment Score
def analyze_trailer_sentiment(comments):
    if pd.isna(comments) or not comments.strip():
        return 0  # Default neutral sentiment
    sentiment_scores = [TextBlob(comment).sentiment.polarity for comment in comments.split('|')]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

df['Trailer_Sentiment_Score'] = df['Trailer_Comments'].apply(analyze_trailer_sentiment)

# New Feature: Pre-Release Hype Score
def calculate_hype_score(trailer_views, social_mentions, google_trends_score):
    # Normalize values and give weight to each component
    weighted_views = trailer_views / 1_000_000 * 0.5  # Scale views in millions
    weighted_mentions = social_mentions / 10_000 * 0.3  # Scale mentions in tens of thousands
    weighted_trends = google_trends_score * 0.2  # Trends score is usually 0-100

    return weighted_views + weighted_mentions + weighted_trends

df['Pre_Release_Hype_Score'] = df.apply(lambda x: calculate_hype_score(
    x['Trailer_Views'], x['Social_Media_Mentions'], x['Google_Trends_Score']
) if not pd.isna(x['Trailer_Views']) else 0, axis=1)



# New Feature: Critic Review Sentiment Score
def analyze_critic_sentiment(reviews):
    if pd.isna(reviews) or not reviews.strip():
        return 0  # Default neutral sentiment
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in reviews.split('|')]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

df['Critic_Review_Sentiment_Score'] = df['Critic_Reviews'].apply(analyze_critic_sentiment)

# New Feature: Audience Review Sentiment Score
def analyze_audience_sentiment(reviews):
    if pd.isna(reviews) or not reviews.strip():
        return 0  # Default neutral sentiment
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in reviews.split('|')]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

df['Audience_Review_Sentiment_Score'] = df['Audience_Reviews'].apply(analyze_audience_sentiment)

# New Feature: Review Volume
def count_reviews(reviews):
    return len(reviews.split('|')) if pd.notna(reviews) and reviews.strip() else 0

df['Review_Volume'] = df['Audience_Reviews'].apply(count_reviews)

# New Feature: Review Sentiment Trend Over Time
def calculate_sentiment_trend(reviews):
    if pd.isna(reviews) or not reviews.strip():
        return 0  # Default neutral trend
    sentiment_scores = [TextBlob(review).sentiment.polarity for review in reviews.split('|')]
    if len(sentiment_scores) < 2:
        return 0  # Not enough data for trend calculation
    return sentiment_scores[-1] - sentiment_scores[0]  # Change in sentiment from first to latest review

df['Review_Sentiment_Trend'] = df['Audience_Reviews'].apply(calculate_sentiment_trend)


# New Feature: Actor's Career Length
def get_actor_career_length(actor):
    # For simplicity, we'll assume we have data on when actors started their careers.
    actor_career_start = {
        'Leonardo DiCaprio': 1991,
        'Robert Downey Jr.': 1970,
        'Meryl Streep': 1977,
        # Add more actors as needed
    }
    return 2025 - actor_career_start.get(actor, 2000)  # Default to 2000 if not found

df['Actor_Career_Length'] = df['Lead_Actor'].apply(get_actor_career_length)
def analyze_genre_sentiment(genres):
    sentiment_scores = []
    for genre in genres.split(","):
        analysis = TextBlob(genre.strip())
        sentiment_scores.append(analysis.sentiment.polarity)
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
def count_genres(genres):
    """
    Count the number of unique genres a movie belongs to.
    """
    return len(set(genres.split(',')))

# New Feature: Actor's Career Length
def get_actor_career_length(actor):
    # For simplicity, we'll assume we have data on when actors started their careers.
    actor_career_start = {
        'Leonardo DiCaprio': 1991,
        'Robert Downey Jr.': 1970,
        'Meryl Streep': 1977,
        # Add more actors as needed
    }
    return 2025 - actor_career_start.get(actor, 2000)  # Default to 2000 if not found

df['Actor_Career_Length'] = df['Lead_Actor'].apply(get_actor_career_length)

# New Feature: Lead Actor's Total Oscar Wins
def get_actor_oscar_wins(actor):
    actor_oscar_wins = {
        'Leonardo DiCaprio': 1,
        'Robert Downey Jr.': 0,
        'Meryl Streep': 3,
        'Denzel Washington': 2,
        'Tom Hanks': 2,
        # Add more actors as needed
    }
    return actor_oscar_wins.get(actor, 0)  # Default to 0 if not found

df['Lead_Actor_Oscar_Wins'] = df['Lead_Actor'].apply(get_actor_oscar_wins)

# New Feature: Lead Actor's Average Box Office Earnings
def get_actor_avg_box_office(actor):
    actor_box_office = {
        'Leonardo DiCaprio': 400_000_000,
        'Robert Downey Jr.': 750_000_000,
        'Meryl Streep': 180_000_000,
        'Denzel Washington': 220_000_000,
        'Tom Hanks': 300_000_000,
        # Add more actors as needed
    }
    return actor_box_office.get(actor, 100_000_000)  # Default to $100M if not found

df['Lead_Actor_Avg_Box_Office'] = df['Lead_Actor'].apply(get_actor_avg_box_office)

# New Feature: Director's Oscar Wins
def get_director_oscar_wins(director):
    director_oscar_wins = {
        'Steven Spielberg': 3,
        'Christopher Nolan': 1,
        'Martin Scorsese': 1,
        'Quentin Tarantino': 2,
        'James Cameron': 1,
        # Add more directors as needed
    }
    return director_oscar_wins.get(director, 0)  # Default to 0 if not found

df['Director_Oscar_Wins'] = df['Director'].apply(get_director_oscar_wins)

# New Feature: Extract Total Award Nominations
def extract_award_nominations(awards):
    if pd.isna(awards):
        return 0
    import re
    numbers = [int(num) for num in re.findall(r'\d+', awards)]
    return numbers[-1] if numbers else 0  # Assuming last number represents nominations

df['Total_Award_Nominations'] = df['Awards'].apply(extract_award_nominations)

# New Feature: Average IMDb Rating of Director's Past Movies
def get_director_avg_imdb(director):
    past_movies = df[df['Director'] == director]
    return past_movies['Rating'].mean() if not past_movies.empty else 5.0  # Default rating

df['Director_Avg_IMDb_Rating'] = df['Director'].apply(get_director_avg_imdb)


def extract_awards_count(awards):
    if pd.isna(awards): return 0
    import re
    numbers = [int(num) for num in re.findall(r'\d+', awards)]
    return sum(numbers)

df['Awards_Won'] = df['Awards'].apply(extract_awards_count)

# Existing Features
df['Budget'] = df['Budget'].fillna(0).astype(float)
df['Revenue'] = df['Revenue'].fillna(0).astype(float)
df['Budget_to_Revenue_Ratio'] = df.apply(lambda x: x['Budget'] / x['Revenue'] if x['Revenue'] > 0 else 0, axis=1)
df['Director_Name_Length'] = df['Director'].apply(lambda x: len(x) if pd.notna(x) else 0)

# New Feature: Average Runtime per Director
df['Runtime'] = df['Runtime'].fillna('0 min').apply(lambda x: int(x.split()[0]) if isinstance(x, str) else 0)
df['Director_Avg_Runtime'] = df.groupby('Director')['Runtime'].transform('mean')

# New Feature: Number of Genres per Movie
df['Num_Genres'] = df['Genre'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)

# New Feature: Number of Words in Title
df['Title_Word_Count'] = df['Title'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)

# New Feature: Sentiment Analysis of Movie Titles
df['Title_Sentiment'] = df['Title'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

# New Feature: Lead Actor Popularity
def get_lead_actor(actors):
    return actors.split(',')[0] if pd.notna(actors) else 'Unknown'

def get_actor_popularity(actor):
    actor_popularity = {'Leonardo DiCaprio': 90, 'Robert Downey Jr.': 85, 'Meryl Streep': 88}  # Example data
    return actor_popularity.get(actor, 50)  # Default popularity

df['Lead_Actor'] = df['Actors'].apply(get_lead_actor)
df['Lead_Actor_Popularity'] = df['Lead_Actor'].apply(get_actor_popularity)

# New Feature: Director's Past Success Score
def calculate_director_success(director):
    past_movies = df[df['Director'] == director]
    return past_movies['Rating'].mean() if not past_movies.empty else 5.0  # Default score

df['Director_Success_Score'] = df['Director'].apply(calculate_director_success)

# New Feature: Actor Ensemble Popularity
def calculate_ensemble_popularity(actors):
    actor_list = actors.split(',') if pd.notna(actors) else []
    return np.mean([get_actor_popularity(actor.strip()) for actor in actor_list]) if actor_list else 50

df['Actor_Ensemble_Popularity'] = df['Actors'].apply(calculate_ensemble_popularity)

# New Feature: Average Lead Actor IMDb Rating
def get_lead_actor_rating(actor):
    past_movies = df[df['Lead_Actor'] == actor]
    return past_movies['Rating'].mean() if not past_movies.empty else 5.0

df['Lead_Actor_Avg_IMDb_Rating'] = df['Lead_Actor'].apply(get_lead_actor_rating)

# New Feature: Franchise Popularity Score
def is_franchise(title):
    franchise_keywords = ['Avengers', 'Star Wars', 'Harry Potter', 'Fast & Furious', 'Batman', 'Spider-Man']
    return any(keyword in title for keyword in franchise_keywords)

df['Franchise_Popularity'] = df['Title'].apply(lambda x: 1 if is_franchise(x) else 0)

# Features for prediction
features = ['Year', 'Genre_Sentiment', 'Is_Weekend', 'Is_Holiday_Release', 'Is_Peak_Season',
            'Awards_Won', 'Budget_to_Revenue_Ratio', 'Director_Name_Length', 'Director_Avg_Runtime', 'Num_Genres', 'Title_Word_Count', 'Title_Sentiment', 'Lead_Actor_Popularity', 'Director_Success_Score', 'Actor_Ensemble_Popularity', 'Lead_Actor_Avg_IMDb_Rating', 'Franchise_Popularity']
features += [col for col in df.columns if col.startswith('Season_')]

# X = feature set
X = df[features]

# y = target variable (IMDb Rating)
y = df['Rating'].astype(float)

# Step 4: Normalize the 'Rating' column
scaler = MinMaxScaler()
df['Normalized_Rating'] = scaler.fit_transform(df[['Rating']])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save results
df.to_csv('omdb_top_movies_with_new_features.csv', index=False)
