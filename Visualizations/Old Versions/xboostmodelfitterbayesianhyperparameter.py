import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import optuna
import shap

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file
csv_file_path = os.path.join(script_directory, '../Data/proj_4_feat_target_var_set.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Check the data
print(df.head())

# Specify feature names and types
numerical_features = ['tpc_state_beer_tax_rates_dollar_gal','bea_personal_income_dollars','bea_disp_inc_dollars',	'bea_percapita_personal_income_dollars', 'bea_percapita_disp_inc_dollars',
                      'census_median_household_inc_dollars'] 
percentage_features = ['census_percent_pop_21_plus','census_percent_pop_18_24','census_percent_pop_25_34','census_percent_pop_35_44','census_percent_pop_45_54', 
                       'census_percent_pop_55_64', 'census_percent_pop_65_plus',	'census_percent_pop_male', 'census_percent_pop_female', 'census_percent_pop_married', 'census_percent_pop_widowed',	
                       'census_percent_pop_divorced',	'census_percent_pop_separated',	'census_percent_pop_never_married',	'ed_census_percent_pop_less_hs', 'ed_census_percent_pop_only_hs', 
                       'ed_census_percent_pop_some_college_or_assoc',	'ed_percent_pop_college_grad_only',	'ed_percent_pop_grad_prof_degree', 'census_percent_employed', 'census_percent_unemployed',
                       'census_percent_armed_forces_employment', 'census_percent_not_in_labor_force',	'brfss_drinking_culture_surrogate_metric_percent_binge', 'census_percent_pop_in_poverty_est']  # Replace with your percentage feature names
binary_features = ['high_tourist_border_sales']
target_column = 'niaaa_legal_adult_per_capita_beer_consumed_gallons'

X = df[numerical_features + percentage_features + binary_features]
y = df[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('pct', MinMaxScaler(), percentage_features),
        ('cat', 'passthrough', binary_features)
    ],
    remainder='passthrough'  # This ensures any unspecified columns are passed through
)

# Define a function that creates a pipeline with given hyperparameters.
def create_pipeline(n_estimators, learning_rate, max_depth, subsample, colsample_bytree):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42))
    ])
    return pipeline

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    
    pipeline = create_pipeline(n_estimators, learning_rate, max_depth, subsample, colsample_bytree)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Trial {trial.number}, MSE: {mse}, R2: {r2}")
    
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=250)

print("Best Parameters:", study.best_params)
print("Best Mean Squared Error:", study.best_value)

# To apply the best parameters, use the create_pipeline function with the best parameters found
best_pipeline = create_pipeline(**study.best_params)
best_pipeline.fit(X_train, y_train)

# Use the best pipeline to make predictions on the test set
y_pred = best_pipeline.predict(X_test)

# Calculate MSE and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the MSE and R2 score
print(f"Mean Squared Error (MSE) on Test Set: {mse}")
print(f"R^2 Score on Test Set: {r2}")

# Getting feature importances
feature_importances = best_pipeline.named_steps['regressor'].feature_importances_
transformed_feature_names = numerical_features + percentage_features + binary_features
importances_df = pd.DataFrame({
    'Feature': transformed_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importances_df)

# Transform the datasets and create DataFrames with correct feature names
transformed_feature_names = numerical_features + percentage_features + binary_features

X_train_transformed = best_pipeline.named_steps['preprocessor'].transform(X_train)
X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)

# Convert the transformed data back to DataFrames to retain feature names
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names)

# Initialize the SHAP Explainer with the model using the DataFrame (which includes feature names)
explainer = shap.Explainer(best_pipeline.named_steps['regressor'], X_train_transformed_df)

# Calculate SHAP values for the test set transformed DataFrame
shap_values = explainer(X_test_transformed_df)

# Summary plot with correct feature names
shap.summary_plot(shap_values, X_test_transformed_df, feature_names=transformed_feature_names)