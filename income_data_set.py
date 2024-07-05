# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:54:52 2023

@author: HARSHITHA
"""

import pandas as pd
import numpy as np

#taking a named a and passing the column names to the list
a=['age','workclass','fnlwgt','education','education_num','marital-status','occupation','relationship','race','sex','capotal_gain','capital_loss','hours-per-week','naive_country','income']

#assigning the list a to the column namex
data=pd.read_csv(r"C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\income.csv",names=a)

data['workclass'].value_counts()
data['workclass'].unique()
#checking if the value of a rows in a column is '?' and replacing it with the mode value of that column
for col in data.columns:
    for i in range(len(data[col])):
        if isinstance(data[col][i], str) and data[col][i].strip().startswith('?'):
            data[col][i] = pd.NA
mode_values=data.mode().iloc[0]          
data.fillna(mode_values,inplace=True) 
#another method to replace
for col in data.columns:
    data[col].replace(to_replace=' ?',value=data[col].mode()[0],inplace=True)

#converting the string tyoe data into numerical data
s=['workclass','education','marital-status','occupation','relationship','race','sex','naive_country']
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in s:
    data[i]=le.fit_transform(data[i])

#separating the independent and dependent value by converting them into arrray 
x=np.array(data.iloc[ : , :-1])
y=data.iloc[ : ,-1]

#taking 80% data for traing
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)

#importing the knn algorthim and giving the k size=10 to increase the accuracy
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=13)
model.fit(xtrain,ytrain)

#predicting y value for the testig data
ypred=model.predict(xtest)

#getting the accuracy by comparing the ytest and ypred
from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100

#checking the model using some different x values and x values already in the dataframe
#print(model.predict([[33,6,2671,9,12,4,0,1,4,1,25,0,39,38]]))
#print(model.predict([[20,3,17530,15,9,2,3,0,2,1,0,0,20,38]]))
#print(model.predict([[33,0,16257,9,12,1,3,1,4,1,0,0,54,38]]))
#print(model.predict([[65,3,72741,4,16,3,9,2,2,1,82354,0,56,15]]))
import pickle
model_path = r"C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\incomeprediction_savedmodel.sav"
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_path}")
