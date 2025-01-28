# Add a new feature for Production Budget Scale
def classify_budget(budget):
    """Classify movies into budget categories based on production cost."""
    if budget < 10:
        return 'Low'
    elif 10 <= budget < 50:
        return 'Medium'
    else:
        return 'High'

# Apply the function to create a new column for budget scale
df['Budget_Scale'] = df['Production_Budget_Millions'].apply(classify_budget)

# One-hot encode the 'Budget_Scale' feature to treat it as categorical data
df = pd.get_dummies(df, columns=['Budget_Scale'], drop_first=True)

# Update the features list to include the new 'Budget_Scale' columns
features += [col for col in df.columns if col.startswith('Budget_Scale_')]

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
