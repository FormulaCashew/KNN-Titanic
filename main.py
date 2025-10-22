import os
from platform import processor

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Imputation import DataFrameImputation
from KNN_Model import KNN

# Define
impute_data = True
path_to_df = "../../Datasets/titanic.csv"

def plot_confusion_matrix(y_true, y_pred):
    """
    Generates, displays and saves a confusion matrix heatmap from the data given
    Args:
        y_true (array-like): A list of the true values
        y_pred (array-like): A list of the predicted values
    """
    # Confusion matrix
    correlation_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data=correlation_mat,
                annot=True,
                fmt="d",
                cmap="GnBu",
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig("./confusion_matrix.png", dpi=400)

def fill_data(df: pd.DataFrame, exclusions: list = None) -> pd.DataFrame:
    """
    Simple function to fill missing values in a dataframe by doing an imputation
    Uses DataFrameImputation class which is a local class made for this code, should be able to work with other dataframes
    Args:
        df (pd.DataFrame): Dataframe to fill
        exclusions (list): List of column names to exclude
    Returns:
        pd.DataFrame: Dataframe filled
    """

    # Process starting data
    print(df.head())
    if exclusions:
        df = df.drop(columns = exclusions, errors='ignore')
    # Data needs to be imputed
    imputator = DataFrameImputation(df)
    print(imputator.column_resume())
    imputator.plot_distribution(cols=['age', 'fare'], prefix="Before")
    imputator.impute_data(method='knn', k_neighbors=5)
    imputed_df = imputator.get_df()

    # Encode sex
    imputed_df['sex'] = imputed_df['sex'].map({'male': 0, 'female': 1})

    print(imputed_df.head())
    imputator.compare_distributions(imputator.original_df, imputed_df, columns=['age'], labels=('Before', 'After'), plot_type='hist')
    imputator.compare_distributions(imputator.original_df, imputed_df, columns=['age'], labels = ('Before', 'After'), plot_type='box')
    output_filename = 'titanic_imputed.csv'
    try:
        imputed_df.to_csv(output_filename, index=False)
        print(f"Successfully saved the cleaned data to '{output_filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    return imputed_df

def plot_elbow(x_train, x_test, y_train, y_test, k_max : int = 10):
    # Sphynx style docstring
    """
    Function to calculate, and generate an elbow plot
    :param x_train: Input training data
    :type x_train: pd.DataFrame
    :param x_test: Input test data
    :type x_test: pd.DataFrame
    :param y_train: Output training data
    :type y_train: pd.DataFrame
    :param y_test: Output testing data
    :type y_test: pd.DataFrame
    :param k_max: Maximum number of neighbors to use
    :type k_max: int
    :return optimal_k: Optimal k value to use
    :rtype optimal_k: int
    """
    errors = []

    knn = KNN(k=1)
    knn.store(x_train, y_train)
    k_range = range(1, k_max)
    for k in k_range:
        knn.set_k(k)
        knn.predict(x_test)
        predictions = knn.predict(x_test)
        accuracy = (np.sum(predictions == np.array(y_test)))/len(y_test)
        error = 1 - accuracy
        errors.append(error)
        print(f"Error: {error:.4f} with k={k}")

    plt.figure(figsize=(10, 10))
    plt.plot(k_range, errors, label='Elbow method', marker='o')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Error')
    plt.xticks(k_range)
    plt.legend()
    plt.savefig("./elbow_error.png", dpi=400)
    plt.show()
    return errors.index(min(errors))


def main():
    init_df = pd.read_csv(path_to_df)

    init_df.head()
    # Columns to drop
    exclusions = ['deck', 'embark_town', 'boat', 'body', 'home.dest']

    #imputation
    if impute_data:
        if os.path.exists("./titanic_imputed.csv"):
            titanic_df = pd.read_csv("./titanic_imputed.csv")  # load from existing file
        else:
            titanic_df = fill_data(init_df, exclusions)  # when there is incomplete data and load it
    else:
        # Without imputation data
        titanic_df = pd.read_csv(path_to_df)
        titanic_df['sex'] = titanic_df['sex'].map({'male': 0, 'female': 1})
        titanic_df.dropna()
        titanic_df.drop(columns=exclusions, errors='ignore')

    # The main features are: age, fare, pclass, sex, sibsp, parch
    features = ['age', 'fare', 'pclass', 'sex', 'sibsp', 'parch']
    to_find = 'survived'

    inputs = titanic_df[features]
    outputs = titanic_df[to_find]

    # Normalize only the inputs
    process = DataFrameImputation(inputs)
    process.normalize()
    normalized_inputs = process.get_df()

    # Split data to test for precision
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(normalized_inputs, outputs, test_size=0.2, random_state=42, stratify = outputs)

    knn_model : KNN = KNN(k=13)
    knn_model.store(inputs_train, outputs_train)

    predictions = knn_model.predict(inputs_test)
    correct_predictions = np.sum(predictions == np.array(outputs_test))
    accuracy = correct_predictions / len(outputs_test)
    print(f"Accuracy on the test set: {accuracy:.3f}")

    plot_confusion_matrix(outputs_test, predictions)
    plot_elbow(inputs_train, inputs_test, outputs_train, outputs_test, k_max = 20)

if __name__ == '__main__':
    main()