from platform import processor

import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Imputation import DataFrameImputation
from KNN_Model import KNN

def fill_data(df: pd.DataFrame, exclusions: list = None) -> pd.DataFrame:

    """
    Simple function to fill missing values in a dataframe by doing an imputation
    Args:
        df (pd.DataFrame): Dataframe to fill
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

def main():
    path_to_df = "../../Datasets/titanic.csv"
    init_df = pd.read_csv(path_to_df)

    #imputation
    exclusions = ['deck', 'embark_town', 'boat', 'body', 'home.dest']
    titanic_df = fill_data(init_df, exclusions) # when there is incomplete data

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


    knn_model : KNN = KNN(k=3)
    knn_model.store(inputs_train, outputs_train)

    predictions = knn_model.predict(inputs_test)
    correct_predictions = np.sum(predictions == np.array(outputs_test))
    accuracy = correct_predictions / len(outputs_test)
    print(f"Accuracy on the test set: {accuracy:.3f}")




if __name__ == '__main__':
    main()