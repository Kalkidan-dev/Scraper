import requests
import pandas as pd
import matplotlib.pyplot as plt

# Your OMDb API key
api_key = '121c5367'

# Function to get movie data from OMDb 
def get_movie_data(title):
    params = {
        't': title,
        'apikey': api_key
    }
    response = requests.get('http://www.omdbapi.com/', params=params)
    return response.json()

# Example list of top-rated movie titles to fetch
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
    if data['Response'] == 'True':
        movie_data.append(data)

# Create DataFrame
df = pd.DataFrame(movie_data)

# Select relevant columns and rename for clarity
df = df[['Title', 'Year', 'imdbRating', 'Genre', 'Director']]
df['Rating'] = df['imdbRating'].astype(float)

# Display the DataFrame
print(df)

# Save to CSV
df.to_csv('omdb_top_movies.csv', index=False)

# Example Rotten Tomatoes data
rt_data = {
    'Title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', '12 Angry Men', 'Schindler\'s List',
              'Pulp Fiction', 'The Lord of the Rings: The Return of the King', 'The Good, the Bad and the Ugly',
              'Fight Club', 'Forrest Gump'],
    'RT_Rating': [91, 98, 94, 100, 97, 92, 95, 97, 79, 71]
}
rt_df = pd.DataFrame(rt_data)

# Merge IMDb and Rotten Tomatoes data on the movie title
combined_df = pd.merge(df, rt_df, on='Title', how='inner')

# Correlation between IMDb and Rotten Tomatoes ratings
correlation_matrix = combined_df[['Rating', 'RT_Rating']].astype(float).corr()
print(correlation_matrix)

# Scatter plot to compare IMDb and Rotten Tomatoes ratings
plt.figure(figsize=(10,6))
plt.scatter(combined_df['Rating'].astype(float), combined_df['RT_Rating'].astype(float), alpha=0.5)
plt.xlabel('IMDb Rating')
plt.ylabel('Rotten Tomatoes Rating')
plt.title('Comparison of IMDb and Rotten Tomatoes Ratings')
plt.show()

# Get the top 10 movies
top_10_movies = df.sort_values(by='Rating', ascending=False).head(10)
print(top_10_movies)

# Plot distribution of ratings
plt.hist(df['Rating'], bins=20, edgecolor='k')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of IMDb Ratings')
plt.show()

# Plot top 10 movies by rating
plt.figure(figsize=(10, 6))
plt.barh(top_10_movies['Title'], top_10_movies['Rating'], color='skyblue')
plt.xlabel('IMDb Rating')
plt.ylabel('Movie Title')
plt.title('Top 10 IMDb Movies by Rating')
plt.gca().invert_yaxis()  # To display the highest rating at the top
plt.show()