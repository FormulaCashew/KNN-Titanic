import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

class DataFrameImputation:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize DataFrame imputation
        Args:
            dataframe (pd.DataFrame): Dataframe to be imputed
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.original_df = dataframe.copy()
        self.df = dataframe.copy()

    def get_df(self):
        """ Getter for the dataframe """
        return self.df.copy()

    def reset_df(self):
        """ Resets the dataframe to its original state."""
        self.df = self.original_df.copy()

    def impute_data(self, method: str='mean', k_neighbors: int=5):
        """
        Impute missing values in dataframe
        Args:
            self.df (pd.DataFrame): Dataframe to be imputed
            method (str, optional): Method to use for imputation. Defaults to 'mean'.
            k_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
        """
        # Numeric columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            if method == 'knn':
                # KNN imputation for numeric
                imp_num = KNNImputer(n_neighbors=k_neighbors)
                self.df[num_cols] = imp_num.fit_transform(self.df[num_cols])
            else:
                # SimpleImputer for (mean/median)
                imp_num = SimpleImputer(strategy=method)
                self.df[num_cols] = imp_num.fit_transform(self.df[num_cols])

        # Categorical columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) > 0:
            # SimpleImputer for categorical columns (most frequent)
            imp_cat = SimpleImputer(strategy='most_frequent')
            self.df[cat_cols] = imp_cat.fit_transform(self.df[cat_cols])
        return self

    def normalize(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            scaler = StandardScaler()
            self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        return self

    def column_resume(self) -> pd.DataFrame:
        """
        Generate a summary of the DataFrame Columns , including missing type and missing values.
        Returns:
            pd.DataFrame: Summary of the DataFrame for each column.
        """
        info = []
        for col in self.df.columns:
            missing = self.df[col].isna().sum()
            info.append({
                "Column": col,
                "dtype": str(self.df[col].dtype),
                "Missing": missing,
                "% missing": 100*missing/len(self.df)
            })
        return pd.DataFrame(info).sort_values(by='% missing', ascending=False)

    def plot_distribution(self, cols:list , prefix:str = "", save_path:str=None):
        """
        Generate a histogram and a boxplot of the distribution of the specified values.
        Args:
            cols (list): List of column names.
            prefix (str, optional): Prefix to add to each column. Defaults to "".
            save_path (str, optional): Path to save the figure. Defaults to None.
        """
        for c in cols:
            data = pd.to_numeric(self.df[c], errors='coerce').dropna()
            if data.empty:
                print(f"Skipping plot for '{c}' as it contains NaN data")
                continue
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            sns.boxplot(y=data, ax=axes[0])
            axes[0].set_title(f"{prefix}Boxplot of {c}")
            sns.histplot(data=data, ax=axes[1])
            axes[1].set_title(f"{prefix}Histogram of {c}")
            plt.suptitle(f"{prefix}Distribution of {c}")
            plt.tight_layout(rect=(0.0,0.03,1.0,0.95))

            if save_path is not None:
                file_save_path = f"{save_path}_{c}.png"
                plt.savefig(file_save_path, format="png", dpi=300)
                print("Saved plot to: '{file_path}'")

            plt.show()

    @staticmethod
    def compare_distributions(df1:pd.DataFrame, df2:pd.DataFrame, columns:list, labels:tuple, plot_type:str="hist", save_path:str=None):
        """
        Compare distributions of two dataframes.
        Args:
            df1 (pd.DataFrame): First dataframe to be compared.
            df2 (pd.DataFrame): Second dataframe to be compared.
            columns (list): List of column names.
            labels (tuple): Tuple of column names.
            plot_type (str, optional): Plot type. Defaults to "hist".
            save_path (str, optional): Path to save the figure. Defaults to None.
        """
        for c in columns:
            d1 = pd.to_numeric(df1[c], errors='coerce').dropna()
            d2 = pd.to_numeric(df2[c], errors='coerce').dropna()

            figure, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=(plot_type=="hist"))
            title_prefix = f"{plot_type} Comparison of {c}"

            if plot_type == "hist":
                sns.histplot(data=d1, ax=axs[0], bins=30, alpha=0.7, label=labels[0])
                sns.histplot(data=d2, ax=axs[1], bins=30, alpha=0.7, label=labels[1])
            elif plot_type == "box":
                sns.boxplot(x=d1, ax=axs[0], label=labels[0], width=0.5)
                sns.boxplot(x=d2, ax=axs[1], label=labels[1], width=0.5)
            else:
                raise ValueError(f"Invalid plot type: {plot_type}")

            axs[0].set_title(f"{labels[0]}")
            axs[1].set_title(f"{labels[1]}")

            plt.suptitle(title_prefix, fontsize=14)
            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            if save_path is not None:
                file_save_path = f"{save_path}_{c}.png"
                plt.savefig(file_save_path, format="png", dpi=300)
                print("Saved plot to: '{file_path}'")

            plt.show()
