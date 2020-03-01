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
#train_data,test_data, train_label, test_label = train_test_split(train_data_complete,train_label_complete, random_state =1 )

#Building a decision tree with the K Fold Crossvalidation Method
accuracyDF = pd.DataFrame(columns = ['Max_Depth', 'Accuracy'])

SkV = StratifiedKFold(n_splits = 5, random_state = None)
for i in range(1,30):
    K = 0
    accuracy_aggregate = 0
    for train_index, test_index in SkV.split(train_data_complete,train_label_complete):
        K = K + 1
        train_data = train_data_complete.iloc[train_index]
        train_label = train_label_complete.iloc[train_index]
        test_data = train_data_complete.iloc[test_index]
        test_label = train_label_complete.iloc[test_index]
        DTClassifier = tree.DecisionTreeClassifier(max_depth=i) #model
        DTClassifier.fit(train_data, train_label)     #Building the Model

        predict_label = DTClassifier.predict(test_data) #Making the predictions with the model
        
        #Compare the predicted labels with the test lables for accuracy
        
        accuracy = accuracy_score(test_label, predict_label)
        accuracy_aggregate = accuracy_aggregate + accuracy
    accuracyDF = accuracyDF.append({'Max_Depth':i,'Accuracy':accuracy_aggregate/K},ignore_index = True)
    print('Maximum Depth is %i',i)
    print('Accuracy from K fold is %i', accuracy_aggregate/K)
    #tree.export_graphviz(DTClassifier.tree_  ,out_file='tree.dot', feature_names=train_data.columns)
    
    #call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])

# Plotting the values of accuracy with the maximum depth
accuracyDF.plot(kind= 'scatter',x='Max_Depth',y='Accuracy')

