#importing the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
#import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
validation = pd.read_csv("validation.csv")


# Selecting features and target variable
X_train = train.drop(columns=['Unnamed: 0', 'order_year', 'Delivery_Time'])
y_train = train['Delivery_Time']

X_test = test.drop(columns=['Unnamed: 0', 'order_year', 'Delivery_Time'])
y_test = test['Delivery_Time']

X_val = validation.drop(columns=['Unnamed: 0', 'order_year', 'Delivery_Time'])
y_val = validation['Delivery_Time']

#DECISION TREE REGRESSOR
hp_grid = {
    "max_depth": [3,5,10,15, None],
    "min_samples_split" : [2,5,10,15],
    "min_samples_leaf" : [1,3,5,7]
}
DT = DecisionTreeRegressor(random_state= 30)
grid = GridSearchCV(estimator = DT, param_grid = hp_grid)
grid.fit(X_train, y_train)
print("Best parameters",grid.best_params_)

#prediction

DT_opt = DecisionTreeRegressor(random_state = 29, max_depth = 15, min_samples_leaf =7, min_samples_split = 15 )
DT_opt.fit(X_train, y_train)

#measuring MAE, MSE and R2 score on test data
y_pred = DT_opt.predict(X_test)
print("MAE on test data: %f" % (mean_absolute_error(y_test, y_pred)))
print("MSE on test data: %f" % (mean_squared_error(y_test, y_pred)))
print("R2 on test data: %f" % (r2_score(y_test, y_pred)))