# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:53:28 2023

@author: prate
"""
#Importing necessary libraries

import os
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#Changing directory to root folder 
#Using pandas to load the dataset as a pandas dataframe
#Assigning all "???, "????" values as NaN values 
#Deep copying the dataframe into another datafrane Toyota to work on visualisation without affecting the original data
#Creating dataframe data2 after dropping rows with missing data

os.chdir("D:\\notebooks")
data_income = pd.read_csv("D:\\notebooks\income.csv")                                                                                                         # Additional strings (" ?") to recognize as NA
data = data_income.copy()
data2 = data.dropna(axis=0)


# LOGISTIC REGRESSION

# Reindexing the salary status names to 0,1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

#One-Hot indexing the whole dataframe
new_data=pd.get_dummies(data2, drop_first=True)

# Storing the column names 
columns_list=list(new_data.columns)
print(columns_list)

# Separating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)

# Storing the output values in y
y=new_data['SalStat'].values
print(y)

# Storing the values from input features
x = new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic = LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# Prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Confusion matrix
confusion_matrix = confusion_matrix(test_y, prediction)
print(confusion_matrix)

# Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

# Printing the misclassified values from prediction

print('Misclassified samples: %d' % (test_y != prediction).sum())


# LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES

# Reindexing the salary status names to 0,1
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data3['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data = data3.drop(cols,axis = 1)

new_data=pd.get_dummies(new_data, drop_first=True)

# Storing the column names 
columns_list2=list(new_data.columns)
print(columns_list2)

# Separating the input names from data
features2=list(set(columns_list2)-set(['SalStat']))
print(features2)

# Storing the output values in y
y2=new_data['SalStat'].values
print(y2)

# Storing the values from input features
x2 = new_data[features2].values
print(x2)

# Splitting the data into train and test
train_x2,test_x2,train_y2,test_y2 = train_test_split(x2,y2,test_size=0.3, random_state=0)

# Make an instance of the Model
logistic2 = LogisticRegression()

# Fitting the values for x and y
logistic2.fit(train_x2,train_y2)

# Prediction from test data
prediction2 = logistic2.predict(test_x2)

# Printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y2 != prediction2).sum())
