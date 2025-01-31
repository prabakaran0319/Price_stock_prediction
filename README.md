##Stock Price Prediction Project

#Overview

    This project focuses on predicting stock prices using machine learning techniques.
    The process encompasses data cleaning, exploratory data analysis (EDA), feature engineering,
    model training, and deployment through a Streamlit application.

#Project Workflow

#1.Data Loading and Initial Inspection:

-Loaded the dataset into VS Code.

-Inspected data types and checked for missing values using df.info() and isnull().sum().
 
#2.Data Cleaning:

-Removed empty columns that didn't contribute meaningful 
 information.

-Standardized the date format in the 'Date' column.

-For columns with missing values, filled them with the mean of the 
 respective columns using df[column].fillna(df[column].mean()).

#3.Categorical Data Handling:

-Identified categorical columns by examining unique values.

-Applied one-hot encoding to convert categorical variables into 
 numerical format.

#4.Feature Selection:

-Analyzed correlations between features using a heatmap.

-Removed 11 features that showed low significance in prediction 
 accuracy.

#5.Exploratory Data Analysis (EDA):

-Visualized data distributions and identified outliers using line 
 charts, box plots, histograms, and scatter plots.

#6.Outlier Treatment:

-Addressed outliers using Interquartile Range (IQR) and Z-score 
 methods.

-Verified the removal of outliers by re-plotting box plots.

#7.Data Preparation:

-Reset the index and saved the cleaned dataset as cleaned_data.csv.

-Converted the 'Date' column to Unix timestamp format for modeling 
 purposes.

#8.Model Development:

-Split the data into training and testing sets.

-Applied standard scaling to the features.

-Trained multiple models:

 Linear Regression
 
 Decision Tree Regressor
 
 Random Forest Regressor
 
 XGBoost Regressor

-Evaluated models using metrics such as Mean Absolute Error (MAE), 
 Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² 
 Score.

-Identified Random Forest as the best-performing model based on 
 test RMSE.

#9.Hyperparameter Tuning:

-Conducted Grid Search to optimize model parameters.


-Best parameters identified: {'max_depth': 7, 'n_estimators': 100}.

-Optimized Random Forest Model Evaluation:
 MAE: 4.3562
 MSE: 61.5793
 RMSE: 7.8472
 R² Score: 0.9873

#10.Model Deployment:

-Saved the optimized model and scaler for deployment.

-Developed a Streamlit application to:

-Collect user inputs.

-Scale inputs using the pre-fitted scaler.

-Predict stock prices using the trained model.

-Display the predicted stock close price to the user.

#Conclusion

This project demonstrates a comprehensive approach to stock price prediction, 
from data preprocessing and exploratory analysis to model development and deployment. 
The Streamlit application provides an interactive platform for users to input relevant features 
and receive predicted stock prices based on the trained Random Forest model.
