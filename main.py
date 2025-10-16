from platform import processor

import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd
from Imputation import DataFrameImputation

path_to_df = "../../Datasets/titanic.csv"

def main():
    titanic_df = pd.read_csv(path_to_df)
    print(titanic_df.head())
    # Data needs to be imputed
    imputator = DataFrameImputation(titanic_df)
    print(imputator.column_resume())

    imputator.plot_distribution(cols=['age', 'fare'], prefix="Before")
    imputator.impute_data(method='median')

    imputed_df = imputator.get_df()
    print(imputed_df.head())

    imputator.compare_distributions(imputator.original_df, imputed_df, columns=['age'], labels=('Before', 'After'), plot_type='hist')
    imputator.compare_distributions(imputator.original_df, imputed_df, columns=['age'], labels = ('Before', 'After'), plot_type='box')

    output_filename = 'titanic_imputed.csv'
    try:
        imputed_df.to_csv(output_filename, index=False)
        print(f"Successfully saved the cleaned data to '{output_filename}'")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':
    main()