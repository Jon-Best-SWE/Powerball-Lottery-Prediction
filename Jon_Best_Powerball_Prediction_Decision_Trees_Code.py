# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:02:45 2023

@author: JonBest
"""

# Jon Best
# 7/16/2023
# The purpose of this Python code is to implement both the Decision Tree Classifier, the Decision Tree Regressor, and the Random
# Forest Regressor models with the Information Gain splitting approach to determine accuracy and predict the most likely winning numbers,
# Power Play number, Powerball number, jackpot amount, amount won, city, and store from a history of Powerball results.
 
#***************************************************************************************
# Title: 4 Simple Ways to Split a Decision Tree in Machine Learning (Updated 2023) 
# Author: Sharma, A.
# Date: 2020
# Availability: https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/
#
# Title: A Complete Guide to Data Visualization in Python With Libraries & More
# Author: Ravikiran A S
# Date: 2023
# Availability: https://www.simplilearn.com/tutorials/python-tutorial/data-visualization-in-python
#
# Title: Create your first Text Generator with LSTM in a few minutes
# Author: Editorial Team
# Date: 2020
# Availability: https://towardsai.net/p/deep-learning/create-your-first-text-generator-with-lstm-in-few-minutes#83cd
#
# Title: Decision Tree Algorithm – A Complete Guide
# Author: Saini, A.
# Date: 2021
# Availability: https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/
#
# Title: Decision Trees for Classification and Regression 
# Author: Codecademy Team
# Date: n.d.
# Availability: https://www.codecademy.com/article/mlfun-decision-trees-article
#
# Title: Entropy and Information Gain to Build Decision Trees in Machine Learning
# Author: Ayuta, C.
# Date: 2021
# Availability: https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
#
# Title: Random Forest Regression in Python Explained
# Author: Chakure, A., & Whitfield, B.
# Date: 2023
# Availability: https://builtin.com/data-science/random-forest-python
#
#***************************************************************************************

# Imported libraries include: pandas to develop dataframes, numpy for numeric computation, 
# matplotlib plus seasborn for graphic representation of data, and sklearn for machine learning functions. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Reading 1st CSV file to retrieve the required data
data = pd.read_csv('POWERBALL-from_0001-01-01_to_2023-07-15_MOD.csv')

###############################################################################

# VISUAL GRAPHS FOR DATASET ANALYSIS

#Display a visual graph that shows the frequency of winning for each number.
# Split the winning numbers and convert them to integers.
winning_numbers = data['Winning Numbers'].str.split(' - ')
all_numbers = [number for sublist in winning_numbers for number in sublist]
winning_numbers_int = [int(number) for number in all_numbers]

# Calculate the histogram without plotting.
hist_values, bin_edges = np.histogram(winning_numbers_int, bins=range(1, 71))

# Display a visual graph that shows the frequency of each winning number.
plt.figure(figsize=(18, 8))

# Define custom colors for each bar (alternating blue and light blue in this example).
custom_colors = ['blue', 'lightblue']

# Plot each bar separately with the desired colors.
for i, (value, edge) in enumerate(zip(hist_values, bin_edges[:-1])):
    plt.bar(edge, value, width=1, color=custom_colors[i % 2], edgecolor='black')

plt.xlabel('Number')
plt.ylabel('Frequency')
plt.title('Histogram of Winning Numbers')

# Get the unique winning numbers for labeling the x-axis.
unique_numbers = np.unique(winning_numbers_int)

# Set the xticks and xticklabels for each bar.
plt.xticks(unique_numbers, unique_numbers)

plt.show()
plt.close()

# Display a visual graph that shows the frequency of each Powerball number.
# Count the frequency of each Powerball number.
powerball_counts = data['Powerball'].value_counts().sort_index()

plt.figure(figsize=(18, 8))

# Define custom colors for each bar (alternating blue and light blue in this example).
custom_colors = ['blue', 'lightblue']

# Plot each bar separately with the desired colors.
for i, (value, index) in enumerate(zip(powerball_counts.values, powerball_counts.index)):
    plt.bar(index, value, width=1, color=custom_colors[i % 2], edgecolor='black')

# Set the xticks and xticklabels for each bar (1 through 42).
plt.xticks(range(1, 43))

plt.xlabel('Powerball Number')
plt.ylabel('Frequency')
plt.title('Bar Chart of Powerball')

plt.show()
plt.close()

# Display a visual graph that shows the frequency of each Power Play number.
# Count the frequency of each Powerball number.
powerball_counts = data['Power Play'].value_counts().sort_index()


plt.figure(figsize=(18, 8))

# Define custom colors for each bar (alternating blue and light blue in this example).
custom_colors = ['blue', 'lightblue']

# Plot each bar separately with the desired colors.
for i, (value, index) in enumerate(zip(powerball_counts.values, powerball_counts.index)):
    plt.bar(index, value, width=1, color=custom_colors[i % 2], edgecolor='black')

# Set the xticks and xticklabels for each bar (1 through 42).
plt.xticks(range(1, 11))

plt.xlabel('Power Play Number')
plt.ylabel('Frequency')
plt.title('Bar Chart of Power Play')

plt.show()
plt.close()

# Display a visual graph that shows the jackpot amounts throughout time.
# Convert the date column to datetime.
data['Draw date'] = pd.to_datetime(data['Draw date'])

# Plot the line chart.
plt.figure(figsize=(18, 8))
plt.plot(data['Draw date'], data['Jackpot'])
plt.xlabel('Draw Date')
plt.ylabel('Jackpot Amount')
plt.title('Line Chart of Jackpot Over Time')
plt.xticks(rotation=45)
plt.show()
plt.close()

# Display a scatter plot visual graph that shows a correlation between the jackpot amount and Power Plays.
# Plot the scatter plot.
plt.figure(figsize=(18, 8))
plt.scatter(data['Power Play'], data['Jackpot'])
plt.xlabel('Power Play')
plt.ylabel('Jackpot Amount')
plt.title('Scatter Plot of Power Play vs. Jackpot')
plt.show()
plt.close()

# Display a pie chart that shows the difference between all winning and non-winning jackpots.
# Count the number of jackpot winners and non-winners.
jackpot_winners = data[data['Jackpot Winners'] > 0]
jackpot_non_winners = data[data['Jackpot Winners'] == 0]
labels = ['Winners', 'Non-Winners']
sizes = [len(jackpot_winners), len(jackpot_non_winners)]

# Define custom colors for the pie chart (blue and light blue).
custom_colors = ['orange', 'lightblue']

# Plot the pie chart with custom colors.
plt.figure(figsize=(18, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=custom_colors)
plt.title('Pie Chart of Jackpot Winners vs. Non-Winners')
plt.show()
plt.close()

# Display a horizontal bar graph that shows the differences between all the prize categories.
# Define the top prize categories.
prize_categories = [
    'Jackpot', 'Match 5 Prize', 'Match 5 Prize (with Power Play)',
    'Match 4 + Powerball Prize', 'Match 4 + Powerball Prize (with Power Play)',
    'Match 4 Prize (with Power Play)', 'Match 3 + Powerball Prize',
    'Match 3 + Powerball Prize (with Power Play)', 'Match 3 Prize',
    'Match 2 + Powerball Prize', 'Match 2 + Powerball Prize (with Power Play)',
    'Match 1 + Powerball Prize', 'Match 1 + Powerball Prize (with Power Play)',
    'Match 0 + Powerball Prize', 'Match 0 + Powerball Prize (with Power Play)',
    'Double Play Jackpot', 'Double Play Match 5 Prize',
    'Double Play Match 4 + Powerball Prize', 'Double Play Match 4 Prize',
    'Double Play Match 3 + Powerball Prize', 'Double Play Match 3 Prize',
    'Double Play Match 2 + Powerball Prize', 'Double Play Match 1 + Powerball Prize',
    'Double Play Match 0 + Powerball Prize'
]

# Get the mean prize amount for each category.
mean_prizes = [data[category].mean() for category in prize_categories]

# Define acronyms and full names for each category.
acronyms = [
    'JP', 'M5P', 'M5P w/ PP', 'M4+PB', 'M4+PB w/ PP',
    'M4P w/ PP', 'M3+PB', 'M3+PB w/ PP', 'M3P',
    'M2+PB', 'M2+PB w/ PP', 'M1+PB', 'M1+PB w/ PP',
    'M0+PB', 'M0+PB w/ PP', 'DP JP', 'DP M5P',
    'DP M4+PB', 'DP M4P', 'DP M3+PB', 'DP M3P',
    'DP M2+PB', 'DP M1+PB', 'DP M0+PB'
]

full_names = [
    'Jackpot', 'Match 5 Prize', 'Match 5 Prize (with Power Play)',
    'Match 4 + Powerball Prize', 'Match 4 + Powerball Prize (with Power Play)',
    'Match 4 Prize (with Power Play)', 'Match 3 + Powerball Prize',
    'Match 3 + Powerball Prize (with Power Play)', 'Match 3 Prize',
    'Match 2 + Powerball Prize', 'Match 2 + Powerball Prize (with Power Play)',
    'Match 1 + Powerball Prize', 'Match 1 + Powerball Prize (with Power Play)',
    'Match 0 + Powerball Prize', 'Match 0 + Powerball Prize (with Power Play)',
    'Double Play Jackpot', 'Double Play Match 5 Prize',
    'Double Play Match 4 + Powerball Prize', 'Double Play Match 4 Prize',
    'Double Play Match 3 + Powerball Prize', 'Double Play Match 3 Prize',
    'Double Play Match 2 + Powerball Prize', 'Double Play Match 1 + Powerball Prize',
    'Double Play Match 0 + Powerball Prize'
]

# Define custom colors for each bar.
custom_colors = plt.cm.tab20(np.arange(len(prize_categories)))

# Plot the horizontal bar chart.
plt.figure(figsize=(10, 12))
bars = plt.barh(acronyms, mean_prizes, color=custom_colors)
plt.ylabel('Prize Category (Acronym)')
plt.xlabel('Mean Prize Amount')
plt.title('Horizontal Bar Chart of Top Prizes')

# Add the prize amount next to each bar.
for bar, mean_prize in zip(bars, mean_prizes):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'${mean_prize:.2f}',
             va='center', fontsize=10)

# Create a legend underneath the plot showing the full names for the acronyms.
legend_labels = [f'{acronym} = {name}' for acronym, name in zip(acronyms, full_names)]
legend = plt.legend(bars, legend_labels, title='Prize Category', loc='upper center',
                    bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)

# Adjust the plot layout to make space for the legend.
plt.subplots_adjust(bottom=0.2)
plt.show()

# Display a line graph that shows the number jackpot winners throughout time.
# Convert the date column to datetime
data['Draw date'] = pd.to_datetime(data['Draw date'])

# Calculate the cumulative sum of jackpot winners over time.
winners_cumulative = data['Jackpot Winners'].cumsum()

# Plot the line chart.
plt.figure(figsize=(18, 8))
plt.plot(data['Draw date'], winners_cumulative)
plt.xlabel('Draw Date')
plt.ylabel('Cumulative Jackpot Winners')
plt.title('Line Chart of Jackpot Winners Over Time')
plt.xticks(rotation=45)
plt.show()

# Display a visual graph that compares the jackpot amounts to the jackpot cash value payouts.
# Sort the data by 'Draw date'.
data.sort_values(by='Draw date', inplace=True)

# Extract the 'Draw date', 'Jackpot', and 'Jackpot Cash Value' columns.
draw_dates = data['Draw date']
jackpot_values = data['Jackpot']
cash_values = data['Jackpot Cash Value']

# Create the line graph.
plt.figure(figsize=(18, 8))
plt.plot(draw_dates, jackpot_values, label='Jackpot')
plt.plot(draw_dates, cash_values, label='Jackpot Cash Value')

# Set the title and axis labels.
plt.title('Comparison of Jackpot and Jackpot Cash Value Over Time')
plt.xlabel('Draw Date')
plt.ylabel('Amount')

# Add a legend.
plt.legend()

# Show the line graph.
plt.show()

# Display line graph that shows a distribution of winnering jackpots vs non-winning jackpots over time. 
# Convert 'Draw date' to datetime format.
data['Draw date'] = pd.to_datetime(data['Draw date'])

# Create a new column to indicate winning jackpots.
data['Winning Jackpot'] = data['Jackpot Winners'] > 0

# Group the data by the 'Draw date' and 'Winning Jackpot' columns and calculate the average jackpot amount for each group.
average_jackpot_by_date = data.groupby(['Draw date', 'Winning Jackpot'])['Jackpot'].mean().unstack()

# Plot a line graph.
plt.figure(figsize=(18, 8))
plt.plot(average_jackpot_by_date.index, average_jackpot_by_date[False], label='Non-Winning Jackpots', marker='o')
plt.plot(average_jackpot_by_date.index, average_jackpot_by_date[True], label='Winning Jackpots', marker='o')
plt.xlabel('Draw Date')
plt.ylabel('Average Jackpot Amount')
plt.title('Average Jackpot Amount Over Time - Winning vs. Non-Winning Jackpots')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###############################################################################

# ACCURACY, PRECISION, RECALL, F1-SCORE USING DECISION TREE CLASSIFICATION, 
# RANDOM FOREST REGRESSOR, AND INFORMATION GAIN SPLITTING APPROACH.

# Drop non-numeric columns (date columns).
data = data.drop(columns=['Draw date', 'Last Day To Claim'])

# Define the feature columns.
classification_feature_columns = [
    'Powerball', 'Power Play', 'Jackpot Winners', 'Jackpot CO Winners'
]

regression_feature_columns = [
    'Jackpot', 'Jackpot Cash Value'
]

# Separate features and labels for classification and regression.
classification_labels = data['Winning Numbers']
classification_features = data[classification_feature_columns].copy()

regression_labels = data['Jackpot']
regression_features = data[regression_feature_columns].copy()

# Handle missing values for both classification and regression features.
classification_features.fillna(classification_features.mean(), inplace=True)
regression_features.fillna(regression_features.mean(), inplace=True)

# Scale the classification features to ensure they are within a similar range.
scaler_classification = StandardScaler()
classification_features_scaled = scaler_classification.fit_transform(classification_features)

# Split dataset into training and testing subsets for classification.
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    classification_features_scaled, classification_labels, test_size=0.2, random_state=42
)

# Create Decision Tree classifier for classification.
decision_tree_class = DecisionTreeClassifier(criterion='entropy')
decision_tree_class.fit(X_train_class, y_train_class)

# Make predictions on test data for classification.
y_pred_class = decision_tree_class.predict(X_test_class)

# Evaluate accuracy, precision, recall, and F1-score for classification.
accuracy_class = accuracy_score(y_test_class, y_pred_class)
precision_class = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
recall_class = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
f1_class = f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)

print('Classification Accuracy:', accuracy_class)
print('Classification Precision:', precision_class)
print('Classification Recall:', recall_class)
print('Classification F1-score:', f1_class)

###############################################################################

# DETERMINE MOST COMMON WINNING NUMBERS, POWERBALL NUMBERS, AND POWER PLAY NUMBERS.

# Extract winning numbers and split into individual numbers.
winning_numbers = data['Winning Numbers'].str.split(' - ')

# Flatten the list of winning numbers.
all_numbers = [number for sublist in winning_numbers for number in sublist]

# Convert winning numbers to integers.
winning_numbers_int = [int(number) for number in all_numbers]

# Count the occurrence of each winning number.
winning_numbers_counts = pd.Series(winning_numbers_int).value_counts()

# Get the most common winning numbers.
most_common_winning_numbers = winning_numbers_counts.head(10)

print("Most common winning numbers:")
print(most_common_winning_numbers)

# Extract Powerball numbers and exclude missing or invalid values.
powerball_numbers = data['Powerball']
powerball_numbers = powerball_numbers[powerball_numbers.notna()]  

# Convert Powerball numbers to integers.
powerball_numbers_int = powerball_numbers.astype(int)

# Count the occurrence of each Powerball number.
powerball_numbers_counts = powerball_numbers_int.value_counts()

# Get the most common Powerball numbers.
most_common_powerball_numbers = powerball_numbers_counts.head(10)  

print("Most common Powerball numbers:")
print(most_common_powerball_numbers)

# Extract Power Play numbers and exclude missing or invalid values.
powerplay_numbers = data['Power Play']
powerplay_numbers = powerplay_numbers[powerplay_numbers.notna()] 

# Convert Power Play numbers to integers.
powerplay_numbers_int = powerplay_numbers.astype(int)

# Count the occurrence of each Power Play number.
powerplay_numbers_counts = powerplay_numbers_int.value_counts()

# Get the most common Power Play numbers.
most_common_powerplay_numbers = powerplay_numbers_counts.head(10)

print("Most common Power Play numbers:")
print(most_common_powerplay_numbers)

###############################################################################

# PREDICT THE MOST LIKELY WINNING NUMBERS, POWER PLAY NUMBER, POWERBALL NUMBER, 
# JACKPOT, AMOUNT WON, AND CITY/STORE THE TICKET WAS PURCHASED.

# Predict on new, unseen data for classification.
new_data_class = pd.DataFrame({
    'Powerball': [20],
    'Power Play': [3],
    'Jackpot Winners': [0],
    'Jackpot CO Winners': [0]
})

# Handle missing values in the new data for classification.
new_data_class.fillna(classification_features.mean(), inplace=True)
new_data_class_scaled = scaler_classification.transform(new_data_class)

# Make predictions on the scaled new data for classification.
new_predictions_class = decision_tree_class.predict(new_data_class_scaled)

print('Predicted winning numbers:')
print(new_predictions_class)

# Extract the Power Play and Powerball numbers from the new data.
new_power_play = new_data_class['Power Play'].iloc[0]
new_powerball = new_data_class['Powerball'].iloc[0]

print('Predicted Power Play number:', new_power_play)
print('Predicted Powerball number:', new_powerball)


# Scale the regression features to ensure they are within a similar range.
scaler_regression = StandardScaler()
regression_features_scaled = scaler_regression.fit_transform(regression_features)

# Split dataset into training and testing subsets for regression.
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    regression_features_scaled, regression_labels, test_size=0.2, random_state=42
)

# Create Decision Tree regressor for regression.
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train_reg, y_train_reg)  # Fit the model with training data

# Access feature importances (information gain values) for regression.
information_gains = decision_tree_reg.feature_importances_
print('Information Gains:', information_gains)

# Make predictions on test data for regression.
y_pred_reg = decision_tree_reg.predict(X_test_reg)

# Evaluate R-squared for regression.
r_squared = r2_score(y_test_reg, y_pred_reg)
print('Regression R-squared:', r_squared)

# Predict on new, unseen data for regression.
new_data_reg = pd.DataFrame({
    'Jackpot': [750000000],
    'Jackpot Cash Value': [378800000]
})

# Handle missing values in the new data for regression.
new_data_reg.fillna(regression_features.mean(), inplace=True)
new_data_reg_scaled = scaler_regression.transform(new_data_reg)

# Make predictions on the scaled new data for regression.
predicted_jackpot = decision_tree_reg.predict(new_data_reg_scaled)[0] 

print('Predicted Jackpot Amount:', predicted_jackpot)

###############################################################################

# Reading 2nd CSV file to retrieve the required data.
data2 = pd.read_csv('powerball_winners.csv')

# Drop unnecessary columns.
data2 = data2.drop(columns=['Game', 'Winner Name', 'Date Won'])

# Convert categorical features to strings.
data2['City'] = data2['City'].astype(str)
data2['Store'] = data2['Store'].astype(str)

# Use label encoding for categorical features.
label_encoder_city = LabelEncoder()
label_encoder_store = LabelEncoder()

data2['City'] = label_encoder_city.fit_transform(data2['City'])
data2['Store'] = label_encoder_store.fit_transform(data2['Store'])

# Separate features and labels.
X = data2.drop(columns=['Amount Won'])
y = data2['Amount Won']

# Create Random Forest Regressor.
random_forest_reg = RandomForestRegressor(n_estimators=100)
random_forest_reg.fit(X, y)

# Predict on new, unseen data.
new_data = pd.DataFrame({
    'City': ['LAKEWOOD'],
    'Store': ['CIRCLE K #2709884']
})

# Use the same encoder for label encoding on the new data.
new_data['City'] = label_encoder_city.transform(new_data['City'])
new_data['Store'] = label_encoder_store.transform(new_data['Store'])

# Make predictions on the new data.
predicted_amount_won = random_forest_reg.predict(new_data)[0]

# Inverse transform label encoding to get predicted 'City' and 'Store'.
predicted_city = label_encoder_city.inverse_transform(new_data['City'])[0]
predicted_store = label_encoder_store.inverse_transform(new_data['Store'])[0]

print('Predicted Amount Won:', predicted_amount_won)
print('Predicted City:', predicted_city)
print('Predicted Store:', predicted_store)
