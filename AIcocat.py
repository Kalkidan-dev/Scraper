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

# List of top actors
top_actors = [
    "Morgan Freeman", "Marlon Brando", "Christian Bale", "Al Pacino",
    "Leonardo DiCaprio", "Tom Hanks", "Robert De Niro", "Brad Pitt",
    "Johnny Depp", "Denzel Washington"
]

# Function to get movie data from OMDb with retry logic
def get_movie_data(title, retries=3, delay=5):
    params = {'t': title, 'apikey': api_key}
    for attempt in range(retries):
        try:
            response = requests.get('http://www.omdbapi.com/', params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('Response') == 'True':
                runtime_str = data.get('Runtime', '0 min')
                try:
                    runtime = int(runtime_str.split()[0])
                except ValueError:
                    runtime = 0
                data['Runtime_Minutes'] = runtime

                # Extract the number of awards won
                awards_str = data.get('Awards', '')
                data['Awards_Won'] = extract_awards_won(awards_str)

                # Count top actors
                actors_str = data.get('Actors', '')
                data['Top_Actors_Count'] = count_top_actors(actors_str)
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

# Function to extract the number of awards won
def extract_awards_won(awards_str):
    """Parse the 'Awards' string to count total awards won."""
    try:
        words = awards_str.split()
        won = 0
        for i, word in enumerate(words):
            if word.lower() in ['win', 'wins']:
                won += int(words[i - 1])  # The number preceding "win/wins"
            elif word.lower() in ['award', 'awards']:
                won += int(words[i - 1])  # The number preceding "award/awards"
        return won
    except (ValueError, IndexError):
        return 0  # Default to 0 if parsing fails

# Function to count the number of top actors in a movie
def count_top_actors(actors_str):
    """Count how many actors in the 'Actors' string are in the top actors list."""
    actors = [actor.strip() for actor in actors_str.split(",")]
    return sum(1 for actor in actors if actor in top_actors)

# Function to analyze the sentiment of the movie genre
def analyze_genre_sentiment(genre):
    sentiment_score = analyzer.polarity_scores(genre)
    return sentiment_score['compound']

# Function to create a holiday release indicator
def is_holiday_release(date):
    holiday_dates = ['12-25', '11-26']
    return int(date.strftime('%m-%d') in holiday_dates)

# Function to calculate the movie's age
def calculate_movie_age(release_date):
    """Calculate the movie's age in years from the current date."""
    if pd.isnull(release_date):
        return 0
    return datetime.now().year - release_date.year

# Function to convert Rotten Tomatoes score to sentiment
def get_rt_sentiment(rt_score_str):
    """Convert Rotten Tomatoes score to sentiment."""
    try:
        rt_score = int(rt_score_str.strip('%'))  # Strip the '%' and convert to integer
        if rt_score >= 75:
            return 1  # Positive sentiment
        elif rt_score >= 50:
            return 0  # Neutral sentiment
        else:
            return -1  # Negative sentiment
    except (ValueError, TypeError):
        return 0  # Default neutral if invalid or missing score

# Example Rotten Tomatoes scores for the movie titles
rotten_tomatoes_scores = {
    'The Shawshank Redemption': '91%',
    'The Godfather': '98%',
    'The Dark Knight': '94%',
    '12 Angry Men': '100%',
    'Schindler\'s List': '98%',
    'Pulp Fiction': '92%',
    'The Lord of the Rings: The Return of the King': '93%',
    'The Good, the Bad and the Ugly': '97%',
    'Fight Club': '79%',
    'Forrest Gump': '71%'
}

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
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Released', 
         'Runtime_Minutes', 'Awards_Won', 'Top_Actors_Count']]
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

# Add the "Rotten Tomatoes Sentiment" feature
df['Rotten_Tomatoes_Score'] = df['Title'].map(rotten_tomatoes_scores).fillna('0%')
df['RT_Sentiment'] = df['Rotten_Tomatoes_Score'].apply(get_rt_sentiment)

# Add the "Budget" feature
df['Budget'] = df['Title'].map({
    'The Shawshank Redemption': 25,
    'The Godfather': 6,
    # Additional movies here
}).fillna(10)

# Add "Movie Age" feature
df['Movie_Age'] = df['Release_Date'].apply(calculate_movie_age)

# Ensure all relevant features are selected
features = ['Year', 'Genre_Sentiment', 'Is_Holiday_Release', 'Is_Weekend', 
            'Runtime_Minutes', 'RT_Sentiment', 'Awards_Won', 'Top_Actors_Count',
            'Movie_Age']

# Extract features and target variable
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

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the actual vs predicted ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Movie Ratings')
plt.show()
