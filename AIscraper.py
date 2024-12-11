import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Assuming your DataFrame `df` and target variable `y` are defined

# Function to simulate social media buzz (you can replace this with actual data)
def generate_social_media_buzz(title):
    """
    Simulate social media buzz as a random number.
    Replace this with real data from a social media API if available.
    """
    return random.randint(1000, 1000000)  # Simulate mentions between 1k and 1M

# Add Social Media Buzz feature
df['Social_Media_Buzz'] = df['Title'].apply(generate_social_media_buzz)

# Add the new feature to the existing feature list
features = [
    'Year', 'Director_Popularity', 'Runtime', 'Budget', 'Movie_Popularity',
    'Genre_Sentiment', 'BoxOffice', 'Awards_Count', 'Genre_Diversity',
    'Release_Month_Sentiment', 'Weekend_Release', 'Sequel', 
    'Critic_Reviews_Sentiment', 'Audience_Engagement_Score', 'Social_Media_Buzz'
]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

# Define the target variable (e.g., 'Rating')
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RandomForestRegressor and GridSearchCV parameters
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearch
best_rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)

# Print evaluation metrics
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R^2: {r2_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

# Visualize Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Regressor - Feature Importance")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
