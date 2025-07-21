Overview
Context:
This is homework 2 of the course 'Become a Kaggle Master'.
This course is part of the IASD Master 2 in Machine Learning, Artificial Intelligence, and Data Science program, jointly run by Université Paris Dauphine, ENS and Mines ParisTech.

Goal:
This latest Kaggle competition aims at predicting a crucial financial time series! The objective is to leverage machine learning techniques to accurately forecast a financial variable, essential for making precise trading predictions. Although the dataset is intentionally anonymized, it encompasses approximately 133 features covering diverse aspects of financial markets. Your task is to develop a robust model capable of predicting the target variable based on the provided features.


Key Considerations:
Given the complexity of financial markets, participants are strongly encouraged to guard against overfitting, as it holds significant weight in the final ranking. The ability to generalize your model's predictions beyond the training data is paramount for success in this competition. As you fine-tune your algorithms, keep in mind the overarching goal of producing robust and reliable forecasts that can withstand the dynamic nature of financial markets.


Description
In this Kaggle competition, participants are tasked with forecasting a stationarized and anonymized financial variable, which serves as a crucial metric for asset managers when making investment decisions based on market conditions. The target variable for prediction is labeled as y, termed "to Predict". The dataset includes a series of predictor variables, known as "X variables", which consist of a variety of traditional financial metrics. These are designed to support investment managers in their decision-making processes by providing insights into market dynamics and potential investment opportunities.

Avoiding Overfitting

A critical aspect of this competition is the emphasis on avoiding overfitting, a common pitfall in financial modeling. Given that financial variables are typically highly non-stationary, the competition organizers have taken measures to stationarize the variable to be predicted, assisting participants in developing more robust models. Despite these efforts, the risk of overfitting remains significant. Competitors are strongly advised to pay careful attention to this aspect, ensuring their models generalize well and are capable of performing consistently across different market scenarios.

Evaluation
Mean Squared Error (MSE)

Mean Squared Error (MSE) is a widely used evaluation metric for regression models, which quantifies the average squared difference between the estimated values and the actual value. 

Key Points:

Sensitivity to Outliers: MSE is sensitive to outliers. Large errors are penalized more due to squaring each difference, which can lead to discrepancies in model evaluation if outliers are present.
Units: The units of MSE are the square of the units of the target variable, which can sometimes make interpretation difficult, particularly when comparing the MSE across different datasets.