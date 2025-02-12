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
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director', 'Release_Date', 'Awards', 'Runtime']]
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

def extract_awards_count(awards):
    if pd.isna(awards): return 0
    import re
    numbers = [int(num) for num in re.findall(r'(\d+)', awards)]
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

# Features for prediction
features = ['Year', 'Genre_Sentiment', 'Is_Weekend', 'Is_Holiday_Release', 'Is_Peak_Season',
            'Awards_Won', 'Budget_to_Revenue_Ratio', 'Director_Name_Length', 'Director_Avg_Runtime']
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

# Display the actual vs. predicted ratings
comparison = pd.DataFrame({'Actual Rating': y_test, 'Predicted Rating': y_pred})
print(comparison.head())

# Predict with new feature included
def predict_rating(year, genre_sentiment, is_weekend, is_holiday_release, is_peak_season, awards_won,
                   budget_to_revenue_ratio, director_name_length, director_avg_runtime, season_features):
    features = [year, genre_sentiment, is_weekend, is_holiday_release, is_peak_season,
                awards_won, budget_to_revenue_ratio, director_name_length, director_avg_runtime] + season_features
    return model.predict(np.array([features]))[0]

# Example usage
season_features = [0] * len([col for col in df.columns if col.startswith('Season_')])
predicted_rating = predict_rating(2024, 0.5, 1, 0, 1, 3, 0.75, 12, 120, season_features)

print(f'Predicted Rating for a movie in 2024 with director avg runtime 120 mins: {predicted_rating:.2f}')

# Save results
df.to_csv('omdb_top_movies_with_new_features.csv', index=False)  # New feature 'Director_Avg_Runtime' included
