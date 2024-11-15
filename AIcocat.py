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

                # Extract the number of awards and nominations
                awards_str = data.get('Awards', '')
                num_awards = sum(int(s) for s in awards_str.split() if s.isdigit() and "award" in awards_str.lower())
                num_nominations = sum(int(s) for s in awards_str.split() if s.isdigit() and "nomination" in awards_str.lower())
                
                data['Number_of_Awards'] = num_awards
                data['Number_of_Nominations'] = num_nominations

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
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Released', 'Runtime_Minutes', 'Number_of_Awards', 'Number_of_Nominations', 'Production', 'Actors']]
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

# Add a Director Popularity Score (example scores)
director_popularity = {
    'Frank Darabont': 9,
    'Francis Ford Coppola': 10,
    'Christopher Nolan': 10,
    'Sidney Lumet': 8,
    'Steven Spielberg': 10,
    'Quentin Tarantino': 9,
    'Peter Jackson': 10,
    'Sergio Leone': 9,
    'David Fincher': 8,
    'Robert Zemeckis': 8
}

# Map the popularity scores to the DataFrame
df['Director_Popularity'] = df['Director'].map(director_popularity).fillna(5)  # Fill missing with average score

# Add the new "Budget" feature (in millions of dollars)
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

# Map the budget values to the DataFrame
df['Budget'] = df['Title'].map(budget_data).fillna(10)  # Fill missing with average budget

# Add Production Studio Popularity (example scores)
production_popularity = {
    'Warner Bros.': 10,
    'Paramount Pictures': 9,
    '20th Century Fox': 8,
    'Universal Pictures': 10,
    'Columbia Pictures': 8,
    'Miramax': 7,
    'New Line Cinema': 7,
    'United Artists': 6,
    'DreamWorks': 9,
    'Metro-Goldwyn-Mayer': 7
}

# Map the popularity scores to the DataFrame
df['Production_Studio_Popularity'] = df['Production'].map(production_popularity).fillna(5)  # Fill missing with average score

# Add Main Actor/Actress Popularity (example scores)
actor_popularity = {
    'Morgan Freeman': 10,
    'Marlon Brando': 10,
    'Christian Bale': 9,
    'Henry Fonda': 8,
    'Liam Neeson': 9,
    'John Travolta': 8,
    'Elijah Wood': 8,
    'Clint Eastwood': 10,
    'Brad Pitt': 10,
    'Tom Hanks': 10
}

# Extract the first actor/actress from the 'Actors' field
df['Main_Actor'] = df['Actors'].apply(lambda x: x.split(',')[0] if pd.notna(x) else None)
df['Main_Actor_Popularity'] = df['Main_Actor'].map(actor_popularity).fillna(5)  # Fill missing with average score

# Create a new feature for the release month
df['Release_Month'] = df['Release_Date'].dt.month

# Categorize the release month into seasons
def categorize_month(month):
    if month in [6, 7, 8]:  # Summer months
        return 'Summer'
    elif month in [11, 12]:  # Holiday months
        return 'Holiday'
    else:
        return 'Other'

df['Release_Season'] = df['Release_Month'].apply(categorize_month)

# One-hot encode the 'Release_Season' feature
df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)

# Ensure all expected columns are present in the DataFrame
for col in ['Release_Season_Summer', 'Release_Season_Holiday']:
    if col not in df.columns:
        df[col] = 0  # Add the missing column with default value 0

# Prepare the data for prediction
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Features for the model (including 'Number_of_Awards', 'Number_of_Nominations', 'Production_Studio_Popularity', and 'Main_Actor_Popularity')
features = [
    'Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Is_Weekend', 
    'Runtime_Minutes', 'Director_Popularity', 'Budget', 
    'Release_Month', 'Release_Season_Summer', 'Release_Season_Holiday',
    'Number_of_Awards', 'Number_of_Nominations', 
    'Production_Studio_Popularity', 'Main_Actor_Popularity'
]
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
def predict_rating(year, genre_sentiment, holiday_release, is_weekend, runtime, director_popularity, budget, release_month, summer_season, holiday_season, num_awards, num_nominations, studio_popularity, actor_popularity):
    # Ensure the input is a 2D array with the same number of features as the model
    return model.predict(np.array([[year, genre_sentiment, holiday_release, is_weekend, runtime, director_popularity, budget, release_month, summer_season, holiday_season, num_awards, num_nominations, studio_popularity, actor_popularity]]))[0]

# Example prediction with all 14 features
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100, 12, 0, 1, 5, 10, 8, 9)
print(f'Predicted Rating for a movie in 2024: {predicted_rating}')
