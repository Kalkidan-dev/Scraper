# Add a new feature for the decade in which the movie was released
def determine_decade(year):
    """Classify the movie into a decade based on its release year."""
    return (year // 10) * 10  # E.g., 1994 -> 1990, 2003 -> 2000

# Apply the function to create a new column for the decade
df['Decade'] = df['Year'].apply(determine_decade)

# One-hot encode the 'Decade' feature to treat it as categorical data
df = pd.get_dummies(df, columns=['Decade'], drop_first=True)

# Update the features list to include the new 'Decade' columns
features += [col for col in df.columns if col.startswith('Decade_')]

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

# Example of how the new feature affects prediction
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100)
print(f'Predicted Rating for a movie in 2024: {predicted_rating:.2f}')
