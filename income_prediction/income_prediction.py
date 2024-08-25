
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\HARSHITHA\Downloads\income1.csv")

print(data.shape)
print(data.columns)
data.columns = data.columns.str.strip()

print(data['income'].unique())

##DATA ANALYSIS

#Income Distribution
income_count = data['income'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(income_count, labels=income_count.index, autopct='%1.1f%%', colors=['orange', 'lightblue'])
plt.title('Income Distribution')
plt.show()

#Income Distribution by Gender
gender_income = data.groupby(['sex', 'income']).size().unstack()
gender_income.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'], figsize=(10, 6))
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of People')
plt.show()

#Income Distribution by Education Level
education_income = data.groupby(['education', 'income']).size().unstack()
education_income.plot(kind='bar', stacked=True, color=['salmon', 'lightgreen'], figsize=(12, 8))
plt.title('Income Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of People')
plt.xticks(rotation=45)
plt.show()

#Distribution of Employees with Income > 50K by Workclass
pi=data[data['income']==' >50K']
ecount=pi['workclass'].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(ecount, labels=ecount.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title('Distribution of Employees with Income > 50K by Workclass')
plt.show()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in data.columns:
    data[i]=le.fit_transform(data[i])

x=np.array(data.iloc[ : , :-1])
y=data.iloc[ : ,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=13)
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypred)*100

import pickle
model_path = r"C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\incomeprediction_savedmodel.sav"
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_path}")



# Save the figures in the static/images/ directory
income_count.plot(kind='pie', autopct='%1.1f%%', colors=['orange', 'lightblue'])
plt.title('Income Distribution')
plt.savefig(r'C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\static\images\income_distribution.png')

gender_income.plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of People')
plt.savefig(r'C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\static\images\gender_income_distribution.png')

education_income.plot(kind='bar', stacked=True, color=['salmon', 'lightgreen'])
plt.title('Income Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Number of People')
plt.xticks(rotation=45)
plt.savefig(r'C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\static\images\education_income_distribution.png')

plt.pie(ecount, labels=ecount.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title('Distribution of Employees with Income > 50K by Workclass')
plt.savefig(r'C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\static\images\workclass_income_distribution.png')




