# Titanic Survival Prediction with K-Nearest Neighbors
## Project Overview
This project implements a complete machine learning pipeline to predict passenger survival on the RMS Titanic. The core of the project is a K-Nearest Neighbors (KNN) classifier built in Python.
The workflow handles the entire process: initial data cleaning, imputation of missing values using KNN, feature engineering, model training, and final evaluation.
---
## File Structure
The project is organized into three main Python scripts:
- `main.py`: Main code to process data, make predictions and plot important figures.
- `KNN_Model.py`: Contains the KNN class, an implementation of the K-Nearest Neighbors algorithm. It handles data storage, Euclidean distance calculation, and prediction logic.
- `Imputation.py`: Contains the DataFrameImputation class, a helper utility for all data preprocessing tasks, including imputation and feature normalization.
---
## The Workflow
The main.py script follows the following process:
1. Data Loading & Cleaning: Loads the Titanic dataset and drops columns irrelevant to prediction.
2. Imputation: Uses the DataFrameImputation class to handle missing values (e.g., in the age column) using a KNN strategy. The cleaned data is cached to titanic_imputed.csv to speed up subsequent runs.
3. Feature Engineering: Converts categorical features (sex) into a numerical format suitable for the model.
4. Data Normalization: Scales all numerical features using StandardScaler. This is critical for distance-based algorithms like KNN to prevent features with large ranges from dominating the results.
5. Model Training: Initializes the from-scratch KNN model with the optimal k and "trains" it by storing the processed training data.
6. Prediction & Evaluation: The model predicts survival on the unseen test set. Performance is evaluated by calculating an accuracy score and displaying a Confusion Matrix.
7. Optimal selection: The model does a series of iterations to find the best k value within a range
---
## Results
The model achieves a predictive accuracy of approximately 75-84% on the test set, depending on the random_state of the train-test split. The performance can be visually inspected using the generated Confusion Matrix, which shows the breakdown of true positives, true negatives, false positives, and false negatives.
***
## How to Run
1. Prerequisites
   -   Ensure you have Python 3 installed, along with the required libraries found in `requirements.txt`
       ````bash
       pip install -r requirements.txt
       ````
2. Dataset 
   - Place the titanic.csv dataset file in a directory structure accessible to the script. The default path is `../../Datasets/titanic.csv`. You may need to adjust this path in `main.py`.
3. File Placement
   - Make sure all three Python scripts (`main.py`, `KNN_Model.py`, `Imputation.py`) are in the same directory.
4. Execute the Script
   - Run the main file from your terminal:
```bash
python main.py
```

