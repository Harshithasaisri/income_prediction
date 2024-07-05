# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:41:29 2024

@author: HARSHITHA
"""

import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the saved model
model_path = r"C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\incomeprediction_savedmodel.sav"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Create LabelEncoders for string columns
workclass_le = LabelEncoder()
education_le = LabelEncoder()
marital_status_le = LabelEncoder()
occupation_le = LabelEncoder()
relationship_le = LabelEncoder()
race_le = LabelEncoder()
sex_le = LabelEncoder()
native_country_le = LabelEncoder()

# Fitting LabelEncoders with training data values
data_path = r"C:\Users\HARSHITHA\OneDrive\Documents\income_prediction\income.csv"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capotal_gain', 'capital_loss', 'hours-per-week', 'naive_country', 'income']
data = pd.read_csv(data_path, names=column_names)

# Replace '?' with mode values
for col in data.columns:
    data[col].replace(to_replace=' ?', value=data[col].mode()[0], inplace=True)

string_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'naive_country']
encoders = [workclass_le, education_le, marital_status_le, occupation_le, relationship_le, race_le, sex_le, native_country_le]

for col, encoder in zip(string_columns, encoders):
    data[col] = encoder.fit_transform(data[col])

# Create the main window
root = tk.Tk()
root.title("Income Prediction")
root.geometry("500x650")

# Function to predict income
def predict_income():
    try:
        # Get input values
        age = int(entry_age.get())
        workclass = entry_workclass.get()
        fnlwgt = int(entry_fnlwgt.get())
        education = entry_education.get()
        education_num = int(entry_education_num.get())
        marital_status = entry_marital_status.get()
        occupation = entry_occupation.get()
        relationship = entry_relationship.get()
        race = entry_race.get()
        sex = entry_sex.get()
        capital_gain = int(entry_capital_gain.get())
        capital_loss = int(entry_capital_loss.get())
        hours_per_week = int(entry_hours_per_week.get())
        native_country = entry_native_country.get()

        # Convert string inputs to numerical using LabelEncoders
        workclass = workclass_le.transform([workclass])[0]
        education = education_le.transform([education])[0]
        marital_status = marital_status_le.transform([marital_status])[0]
        occupation = occupation_le.transform([occupation])[0]
        relationship = relationship_le.transform([relationship])[0]
        race = race_le.transform([race])[0]
        sex = sex_le.transform([sex])[0]
        native_country = native_country_le.transform([native_country])[0]

        # Create input array
        input_data = np.array([[age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country]])

        # Predict income
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            result = ">50K"
        else:
            result = "<=50K"

        messagebox.showinfo("Prediction Result", f"The predicted income is: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create labels and entry fields for each feature
features = [
    ("Age", 10), ("Workclass", 40), ("Fnlwgt", 70), ("Education", 100),
    ("Education Num", 130), ("Marital Status", 160), ("Occupation", 190),
    ("Relationship", 220), ("Race", 250), ("Sex", 280), ("Capital Gain", 310),
    ("Capital Loss", 340), ("Hours Per Week", 370), ("Native Country", 400)
]

entries = {}

for feature, y_pos in features:
    label = tk.Label(root, text=feature, font=("Helvetica", 12))
    label.place(x=50, y=y_pos)
    entry = tk.Entry(root, font=("Helvetica", 12))
    entry.place(x=200, y=y_pos)
    entries[feature] = entry

# Assign each entry to a variable
entry_age = entries["Age"]
entry_workclass = entries["Workclass"]
entry_fnlwgt = entries["Fnlwgt"]
entry_education = entries["Education"]
entry_education_num = entries["Education Num"]
entry_marital_status = entries["Marital Status"]
entry_occupation = entries["Occupation"]
entry_relationship = entries["Relationship"]
entry_race = entries["Race"]
entry_sex = entries["Sex"]
entry_capital_gain = entries["Capital Gain"]
entry_capital_loss = entries["Capital Loss"]
entry_hours_per_week = entries["Hours Per Week"]
entry_native_country = entries["Native Country"]

# Create the Predict button
predict_button = tk.Button(root, text="Predict Income", font=("Helvetica", 14), bg="blue", fg="white", command=predict_income)
predict_button.place(x=200, y=450)

# Run the main loop
root.mainloop()
