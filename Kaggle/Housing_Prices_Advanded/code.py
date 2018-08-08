#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:19:41 2018

@author: architkhatri
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing the dataset and dropping columns with more than 15% nan values
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)

#visualizing Sale price 
sns.distplot(dataset.SalePrice, bins = 150, color = 'red')

#finding the correlation among features
correlation = dataset.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(correlation,vmax = 1, vmin = -1, square=True, cmap = 'bwr')


#pick and use correlated features only using above heatmap
correlated_columns = ['LotArea','OverallQual','YearBuilt',
                      'YearRemodAdd','MasVnrArea','BsmtFinSF1','TotalBsmtSF',
                      '1stFlrSF','2ndFlrSF','GrLivArea','FullBath','TotRmsAbvGrd',
                      'Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                      'OpenPorchSF']

#making the new dataset using only best correlated features
dataset_correlated = dataset[correlated_columns]
X = dataset_correlated.iloc[:,:]


#Filling nan values with mean values of the columns and making training set
X = X.fillna(X.mean()).values
y = dataset.iloc[:,[-1]].values


#Scaling the training data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)


#A simple linear regression model to test the score
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
regressor.score(X,y)


#An attempt to make polynomial linear regression to fit better the data
#but it is not worth it because of high number of features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
regressorPoly = LinearRegression()
regressorPoly.fit(X_poly, y)
regressorPoly.score(X_poly,y)


'''
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
y1 = sc_y.fit_transform(y)
svr.fit(X,y1)
'''

#Using Random Forest Regressor to fit the data
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_forest.fit(X,y)
y_pred = random_forest.predict(X[:30]) #sample predictions (for CV maybe hehe)


#Getting Test data for predictions using above Random Forest Regressor
X_test = pd.read_csv('test.csv')
X_test = X_test[correlated_columns]
X_test = X_test.fillna(X_test.mean()).values
X_test = sc_x.fit_transform(X_test)


#Predictions done using Random Forest Regressor
y_test_pred = random_forest.predict(X_test)


#Making a list of indexes for the submissions
y_test_index = []
for i in range(1459):    
    y_test_index.append(i + 1461)


#Creating submissions dictionary and later converting it into csv
submissions = {'Id': y_test_index, 'SalePrice':y_test_pred}
submissions_df = pd.DataFrame(submissions)
submissions_df.to_csv('submissions.csv', encoding = 'utf-8', index=False)

#Thank you :)
