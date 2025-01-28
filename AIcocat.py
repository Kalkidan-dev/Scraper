# Add a new feature for sequels (e.g., 0 for standalone movies, >0 for sequels)
sequel_data = {
    'The Shawshank Redemption': 0,
    'The Godfather': 1,  # First movie in the franchise
    'The Dark Knight': 2,  # Sequel
    '12 Angry Men': 0,
    'Schindler\'s List': 0,
    'Pulp Fiction': 0,
    'The Lord of the Rings: The Return of the King': 3,  # Third movie in the trilogy
    'The Good, the Bad and the Ugly': 3,  # Third in the "Dollars Trilogy"
    'Fight Club': 0,
    'Forrest Gump': 0
}

# Map the sequel values to the DataFrame
df['Number_of_Sequels'] = df['Title'].map(sequel_data).fillna(0)  # Default to standalone movie (0)

# Update the features list to include 'Number_of_Sequels'
features.append('Number_of_Sequels')

# Re-train the Linear Regression model with the updated features
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

# Example prediction with the new feature
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100)
print(f'Predicted Rating for a movie in 2024: {predicted_rating:.2f}')
