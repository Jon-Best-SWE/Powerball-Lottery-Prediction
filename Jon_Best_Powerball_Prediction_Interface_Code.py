# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:02:45 2023

@author: JonBest
"""

# Jon Best
# 7/16/2023
# The purpose of this Python code is to create a user interface where a user 
# can input five different numbers and retrieve a predicted winning percentage.

#***************************************************************************************
# Title: Leveraging AI and Python for Lotto Number Prediction
# Author: Hautin
# Date: n.d.
# Availability: https://medium.com/@huatin/%E5%88%A9%E7%94%A8-ai-%E5%92%8C-python-%E8%BF%9B%E8%A1%8C%E4%B9%90%E9%80%8F%E5%8F%B7%E7%A0%81%E9%A2%84%E6%B5%8B-e12aa03438e2
#
#***************************************************************************************

# Imported libraries include: tkinter to create graphical user interfaces, 
# pandas to develop dataframes, and sklearn for machine learning functions.  
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Reading CSV file to retrieve the required data.
data = pd.read_csv('POWERBALL-from_0001-01-01_to_2023-07-15_MOD.csv')

# Drop non-numeric columns (date columns).
data = data.drop(columns=['Draw date', 'Last Day To Claim'])

# Preprocess the 'Winning Numbers' feature.
data['Winning Numbers'] = data['Winning Numbers'].apply(lambda x: [int(num) for num in x.split('-')])
winning_numbers = data['Winning Numbers'].tolist()

# Convert column names to strings.
data.columns = data.columns.astype(str)

# Separate features and labels.
classification_labels = data['Jackpot Winners']
classification_features = data['Winning Numbers']

# Create a Random Forest classifier for classification.
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(classification_features.tolist(), classification_labels)

# Function to predict the winning percentage.
def predict_percentage():
    try:
        # Get user input for five numbers.
        input_numbers = entry.get()
        user_numbers = [int(num) for num in input_numbers.split('-')]

        # Make a prediction on the user's numbers.
        prediction = random_forest_classifier.predict([user_numbers])[0]

        # Display the predicted winning percentage.
        messagebox.showinfo("Prediction Result", f"Predicted Winning Percentage: {prediction}%")
    except ValueError:
        messagebox.showerror("Error", "Please enter five valid integer numbers separated by hyphens.")

# Create the main GUI window.
root = tk.Tk()
root.title("Winning Percentage Prediction")

# Create an input field for the user to enter five numbers.
entry = tk.Entry(root)

# Create a label for the input field.
label = tk.Label(root, text="Enter five winning numbers separated by hyphens:")

# Create a button to trigger the prediction.
predict_button = tk.Button(root, text="Predict Winning Percentage", command=predict_percentage)

# Add the widgets to the window.
label.pack()
entry.pack()
predict_button.pack()

# Start the main event loop.
root.mainloop()
