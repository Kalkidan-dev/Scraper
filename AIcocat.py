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

    # Assuming the DataFrame has a 'Release_Month' column
    df['Release_Season'] = df['Release_Month'].apply(classify_season)

    # One-hot encode the 'Release_Season' feature
    df = pd.get_dummies(df, columns=['Release_Season'], drop_first=True)

    # Update the features list to include the new 'Release_Season' columns
    features += [col for col in df.columns if col.startswith('Release_Season_')]

    return df, features


# Call the new function without modifying the existing code
df, features = add_release_season(df, features)

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

# Example prediction with the new feature
predicted_rating = predict_rating(2024, 0.5, 1, 1, 120, 9, 100)
print(f'Predicted Rating for a movie in 2024: {predicted_rating:.2f}')
