# Add a new feature for Language Popularity Score
language_popularity = {
    'English': 10,
    'Spanish': 8,
    'French': 7,
    'German': 6,
    'Mandarin': 9,
    'Hindi': 8,
    'Japanese': 7,
    'Korean': 7,
    'Italian': 6,
    'Other': 5
}

# Assuming there's a 'Language' column in the DataFrame
df['Language_Popularity_Score'] = df['Language'].map(language_popularity).fillna(5)  # Default score for 'Other'

# Update the features list to include 'Language_Popularity_Score'
features.append('Language_Popularity_Score')

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
