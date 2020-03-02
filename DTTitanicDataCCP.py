# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:42:54 2020

@author: arunk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:18:57 2020

@author: arunk
"""

#Importing the required DataSets
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
from subprocess import call
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt




#Loading the data into dataframes
train_data_complete = pd.read_csv('train.csv')
test_data_validation = pd.read_csv('test.csv')

#Removing the survival column from the data 
# Removing NAs from the data as model is not built to handle that
train_data_complete = train_data_complete.dropna()
# 1 of n conversion for sex
train_data_complete.loc[train_data_complete['Sex'] == "male", 'male'] = 1
train_data_complete.loc[train_data_complete['Sex'] == "female", 'male'] = 0
train_data_complete.loc[train_data_complete['Sex'] == "male", 'female'] = 0
train_data_complete.loc[train_data_complete['Sex'] == "female", 'female'] = 1
train_label_complete = train_data_complete['Survived']
train_data_complete = train_data_complete.drop('Survived', axis = 1)

#Removing the irrelevant columns from the data(Fare, Ticket, Name)

train_data_complete = train_data_complete.drop(['Sex','Fare','Ticket','Cabin','PassengerId', 'Name','Embarked'], axis = 1)

# Splitting the data using test train split using Train Test Split
train_data,test_data, train_label, test_label = train_test_split(train_data_complete,train_label_complete, random_state =1 )

#Building a decision tree with the K Fold Crossvalidation Method
accuracyDF = pd.DataFrame(columns = ['Alpha', 'Accuracy'])

DTClassifier = tree.DecisionTreeClassifier(random_state=1)
path = DTClassifier.cost_complexity_pruning_path(train_data,train_label)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

for ccp_alpha in ccp_alphas:
    DTClassifier = tree.DecisionTreeClassifier(random_state=1, ccp_alpha = ccp_alpha)
    DTClassifier.fit(train_data, train_label)
    predict_label = DTClassifier.predict(test_data)
    accuracy = accuracy_score(test_label, predict_label)
    accuracyDF = accuracyDF.append({'Alpha':ccp_alpha,'Accuracy':accuracy},ignore_index = True)


accuracyDF.plot(kind= 'line',x='Alpha',y='Accuracy')

    


