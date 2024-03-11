# Craft Brewer Prediction


## Overview

Welcome to our prediction models which consist of two different predictions.  The first prediction is the consumption of beer for the year 2025 by state and nationwide.  The second prediction is the production of beer for the year 2025 by state and nationwide or by breweries.


## Data Sources 
	Consumption
•	https://www.niaaa.nih.gov/sites/default/files/pcyr1970-2021.txt

Production
•	Brewers Association through a membership (2012-2022)


## Data reliability 

The data we collected came from the Tax and Trade Bureau (TTB), the Beer Institute, 21+ Census data by state 2010-2019, and Beverage alcohol on and off-premises price data which was all aggregated by the brewer’s association through a membership. The consumption data we gathered from the national institute on alcohol abuse and alcoholism government site, which we provided the link for at the data sources site.  For this prediction and the team’s previous project we focused only on beer consumption and production.


## Data integrity 

As we collected our data, we did not focus any attention on special attributes, for example we did not take religious or cultural habits to the geographical data sets, which could impact the outcome of beer consumption in a certain area.  Our data sets did include gender and age for consumption that we were interested in analyzing.


## Prediction Models


Our goal for our prediction model is to predict the consumption of 2025 for the United States and a separate model to produce 2025 for the United States.  

We used a Gradient boosting is a machine learning technique used for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. Here's how it works:
1.	Initialization: It starts with a base model (often a simple average of the target variable) to make initial predictions.
2.	Iterative Improvement:
	•	For each iteration, the algorithm computes the residual, which is the difference between the observed and predicted values from the current model.
	•	A new model (usually a decision tree) is then trained to predict these residuals.
	•	This new model is added to the ensemble, with a coefficient called the learning rate (or shrinkage) applied to control the contribution of each new model. This learning rate is a small positive number (e.g., 0.1) that slows down the learning process to 		make the model more robust.
3.	Additive Modeling:
	•	The predictions from the new model are combined with the predictions from the existing ensemble to form updated predictions.
	•	This process is repeated, with each new model focusing on the residuals (errors) left by the previous models.
4.	Stopping Criteria:
	•	This iterative process continues until a specified number of trees are added or no significant improvement can be made on the prediction accuracy.
The main advantages of gradient boosting are its ability to handle different types of data, robustness to outliers, and effectiveness in capturing complex nonlinear patterns in data. However, it can be prone to overfitting if not carefully tuned, and it may require careful selection of parameters, such as the number of iterations, learning rate, and the depth of the trees.

## Contributors

- Eric Janson - Database
- Zachary Cushman - Data pipeline
- Amanda Rolfe - Data display
- Marc Conwell - Data display

Feel free to reach out for any questions or collaboration opportunities.
