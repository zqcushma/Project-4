import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the processed dataset
data = pd.read_csv('../../Data/training_data.csv')

# Selecting the features and target variable
# You might want to select more relevant features for your model
X = data[['year', 'census_total_pop', 'census_percent_employed', 'tpc_state_beer_tax_rates_dollar_gal']]  # Example feature, include others as necessary
y = data['ba_craft_beer_produced_gallons']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = gb_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R^2: {r2}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Predict for the year 2025
# Adjust the features accordingly if your model uses more than the year
total_population_2025 = 300000000  
percent_employed_2025 = 50
tpc_state_beer_tax_rates_dollar_gal = .40
prediction_2025 = gb_model.predict([[2025, total_population_2025, percent_employed_2025, tpc_state_beer_tax_rates_dollar_gal]])
print(f'Predicted craft beer production for 2025: {prediction_2025[0]} gallons')