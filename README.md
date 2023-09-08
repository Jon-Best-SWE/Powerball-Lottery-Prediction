# Powerball Lottery Prediction (Python)
This project involves the development of a Decision Tree algorithm to predict Powerball lottery outcomes. It leverages data analysis techniques for accurate predictions using decision trees and tree splitting.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn (SKLearn)

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Jon-Best-SWE/Powerball-Lottery-Prediction.git
cd Powerball-Lottery-Prediction
```

2. Analyze the Data
The Python script provided in this repository (`Jon_Best_Powerball_Prediction_Decision_Trees_Code.py`) contains code for analyzing the Powerball lottery results dataset. They are not meant to be run directly from the command line but should be executed within a Python environment or Jupyter Notebook.

3. Open in a Python Environment
You can open and run the provided Python scripts in a Python IDE (e.g., Anaconda, Jupyter Notebook) or code editor (e.g., Visual Studio Code).

4. Run the Code
Inside your Python environment, you can execute the code within these scripts to perform the analysis.

5. View the Results
The results and visualizations will be displayed within your Python environment.

Please note that these scripts are designed for data analysis and should not be run as standalone command-line applications.

# Powerball Winning Percentage Prediction (Python)

This Python script utilizes the `tkinter` library to create a simple graphical user interface (GUI) for predicting the winning percentage of Powerball lottery numbers. The script uses a pre-trained Random Forest Classifier to make predictions based on user input.

## How It Works

1. The script reads Powerball data from a CSV file named `POWERBALL-from_0001-01-01_to_2023-07-15_MOD.csv`.

2. Non-numeric columns ('Draw date' and 'Last Day To Claim') are dropped from the dataset to focus on relevant features.

3. The 'Winning Numbers' feature is preprocessed by splitting it into lists of integers to represent the winning numbers for each draw.

4. The column names are converted to strings for compatibility.

5. The script separates the data into features (Winning Numbers) and labels (Jackpot Winners) to train a Random Forest Classifier.

6. A graphical user interface (GUI) is created using `tkinter`. Users are prompted to input five winning numbers separated by hyphens in an input field.

7. When the "Predict Winning Percentage" button is clicked, the script uses the trained classifier to predict the winning percentage based on the user's input.

8. The predicted winning percentage is displayed in a pop-up message box.

## How to Use

To run the script and predict winning percentages:

1. Ensure you have Python installed on your system.

2. Install the required Python libraries (`pandas` and `scikit-learn`) if you haven't already:
```bash
pip install pandas scikit-learn
```
3. Place the `POWERBALL-from_0001-01-01_to_2023-07-15_MOD.csv` file in the same directory as the script.

4. A GUI window will appear. Enter five winning numbers separated by hyphens in the input field.

5. Click the "Predict Winning Percentage" button to see the predicted winning percentage in a pop-up message.

Please note that this script is for educational purposes and uses a simplified approach to predict winning percentages. Actual lottery outcomes are influenced by various factors and are inherently random

