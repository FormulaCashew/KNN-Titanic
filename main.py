import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd
import Imputation as imp

path_to_df = "../../Datasets/titanic.csv"

def main():
    orignal_data = pd.read_csv(path_to_df)
    print(orignal_data.head())
    # Data needs to be imputed

    return

if __name__ == '__main__':
    main()