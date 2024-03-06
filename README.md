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
This model creation function is designed for use with the Keras Tuner, a tool for hyperparameter tuning in TensorFlow, to automate the process of selecting the optimal architecture and settings for a neural network. The function dynamically constructs a sequential model suitable for binary classification tasks, with flexibility in the number of layers, neurons per layer, and the activation functions used. It begins with specifying the input dimensionality (44 features) and allows for the configuration of up to 10 hidden layers, each potentially containing up to 100 neurons. The Keras Tuner decides the precise architecture—specifically, the number of neurons in the first layer, the total number of layers, and the activation function (ReLU, Tanh, or Sigmoid) used across hidden layers. For binary outcomes, it concludes with a sigmoid-activated layer to output probabilities. In contrast, for continuous outputs, a linear activation would be used (though this is pre-configured for binary outputs). The model is compiled with the Adam optimizer and binary cross-entropy loss, making it suitable for binary classification problems. This approach leverages the power of hyperparameter optimization to enhance model performance by systematically exploring a range of configurations to find the most effective model architecture.

## Contributors

- Eric Janson - Database
- Zachary Cushman - Data pipeline
- Amanda Rolfe - Data display
- Marc Conwell - Data display

Feel free to reach out for any questions or collaboration opportunities.
