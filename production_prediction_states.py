import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the processed dataset
data = pd.read_csv('processed_craft_beer_data.csv')

# Ensure the feature selection keeps the data in DataFrame format
X = data[['year', 'census_total_pop', 'census_percent_employed', 'state_id']]  # DataFrame format
y = data['ba_craft_beer_produced_gallons']

# Encoding categorical data and creating a pipeline
categorical_features = ['state_id']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets, making sure X remains a DataFrame
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the pipeline
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R^2: {r2}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# Specify the state and year for prediction
state_to_predict = 'VA'  # Change to the state you want to predict
year_to_predict = 2025  # Change to the year you want to predict

# Find the latest total population for the specific state from the data
latest_population = data[(data['state_id'] == state_to_predict) & (data['year'] == 2021)]['census_total_pop'].values[0]

# Assuming you have a method to forecast or obtain the 'census_percent_employed' for the state in 2025
percent_employed_forecast = 60  # Example placeholder value, replace with your forecast

# Creating the DataFrame for prediction with the fetched population data
example_state_data = pd.DataFrame({
    'year': [year_to_predict],
    'census_total_pop': [latest_population],
    'census_percent_employed': [percent_employed_forecast],
    'state_id': [state_to_predict]
})

# Use the model to predict for the given state and year
prediction_2025 = model.predict(example_state_data)
print(f'Predicted craft beer production for {state_to_predict} in {year_to_predict}: {prediction_2025[0]} gallons')
