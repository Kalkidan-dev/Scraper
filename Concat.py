import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Assuming movie_data is a list where each item is a dictionary with movie details
movie_data = []

# Step 1: Data Retrieval and Cleaning (Assume 'data' is being appended successfully)
if data:  # Only append if data was successfully retrieved
    movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Release_Date']]
df['Rating'] = df['imdbRating'].astype(float)

# Analyze the sentiment of the movie genres
df['Genre_Sentiment'] = df['Genre'].apply(analyze_genre_sentiment)

# Step 2: Prepare Release Date Features
# Convert 'Release_Date' to a datetime object
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')

# Extract the month from the release date
df['Month'] = df['Release_Date'].dt.month

# Create a season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)

# One-hot encode the season feature
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Extract the day of the week
df['Day_of_Week'] = df['Release_Date'].dt.day_name()

# Create a binary feature for weekend vs. weekday
df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)

# Create a holiday release indicator
def is_holiday_release(date):
    # List of specific holiday dates (you can expand or use a more dynamic method)
    holiday_dates = [
        '12-25',  # Christmas
        '11-26',  # Example: Thanksgiving (date changes yearly)
    ]
    if pd.isna(date):
        return 0
    return int(date.strftime('%m-%d') in holiday_dates)

df['Is_Holiday_Release'] = df['Release_Date'].apply(is_holiday_release)

# Define peak season (e.g., June to August)
df['Is_Peak_Season'] = df['Month'].isin([6, 7, 8]).astype(int)

# Step 3: Prepare the data for prediction
df['Year'] = df['Year'].astype(int)
df['Genre_Sentiment'] = df['Genre_Sentiment'].astype(float)

# Features for prediction
features = ['Year', 'Genre_Sentiment', 'Is_Weekend', 'Is_Holiday_Release', 'Is_Peak_Season']
features += [col for col in df.columns if col.startswith('Season_')]

# X = feature set
X = df[features]

# y = target variable (IMDb Rating)
y = df['Rating'].astype(float)

# Step 4: Normalize the 'Rating' column
scaler = MinMaxScaler()
df['Normalized_Rating'] = scaler.fit_transform(df[['Rating']])

# New Feature: Calculate average rating per director
df['Director_Avg_Rating'] = df.groupby('Director')['Rating'].transform('mean')

# New Feature: Calculate average rating per genre
df['Genre_Avg_Rating'] = df.groupby('Genre')['Rating'].transform('mean')

# New Feature: Calculate number of movies released per year
df['Movies_Per_Year'] = df.groupby('Year')['Title'].transform('count')

# New Feature: Calculate the length of each movie title
df['Title_Length'] = df['Title'].apply(len)

# New Feature: Calculate the total number of genres a movie belongs to
df['Total_Genres'] = df['Genre'].apply(lambda x: len(x.split(',')))

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

# Display the actual vs. predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Bonus: Make a prediction for a new movie
def predict_rating(year, genre_sentiment, is_weekend, is_holiday_release, is_peak_season, season_features, director_avg_rating, genre_avg_rating, movies_per_year, title_length, total_genres):
    features = [year, genre_sentiment, is_weekend, is_holiday_release, is_peak_season] + season_features + [director_avg_rating, genre_avg_rating, movies_per_year, title_length, total_genres]
    return model.predict(np.array([features]))[0]

# New Feature: Extract the number of awards won
def extract_awards_count(awards):
    if pd.isna(awards):
        return 0
    import re
    # Extract numbers from phrases like 'Won 3 Oscars' or 'Another 5 wins & 2 nominations'
    numbers = [int(num) for num in re.findall(r'(\d+)', awards)]
    return sum(numbers)

df['Awards_Won'] = df['Awards'].apply(extract_awards_count)

# Add 'Awards_Won' to the features list
features.append('Awards_Won')

# Update the feature set
X = df[features]

# Re-split the data with the new feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retrain the model with the new feature
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate again
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Updated Mean Squared Error: {mse}')
print(f'Updated R-squared: {r2}')
