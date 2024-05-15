import pandas as pd
import numpy as np

class missing_values:
    def analyse_missing_row(df: pd.DataFrame, ) -> pd.DataFrame:
        # To prepare the list of null that is more than 0
        missing_data_summary = df.isnull().sum()[df.isnull().sum() > 0]

        # If there is no missing data
        if len(missing_data_summary) == 0:
            # Print a statement to allow user to understand
            print("No missing value detected from the dataframe above.")
            # prepare an empty dataframe for return of function
            missing_df = pd.DataFrame()
        else: # If missing data detected from the dataframe
            # Generate the missing table data
            missing_data_summary = pd.DataFrame(missing_data_summary).rename(columns = {0:"count"})
            # To prepare the columns name that have missing data
            missing_data_columns = missing_data_summary.index
            # Prepare the data that have missing values
            missing_df = df.query(" | ".join([f"{column}.isnull()" for column in missing_data_columns]))
            # To put the information into the missing_data_summary
            missing_data_summary.loc["All_rows_with_missing_values", "count"] = len(missing_df)
            # Calculate the percentage of missing data
            missing_data_summary.loc[:,"missing_percentage"] = (missing_data_summary.loc[:,"count"] / len(df) * 100).round(2)

            # Print the missing values rows information.
            print(f"Missing data detected for columns {", ".join(missing_data_columns)}.")
            print("Summary of the missing values from the dataframe =")
            print(missing_data_summary.to_markdown(tablefmt = "pretty"))

        # Return the dataframe with missing values
        return missing_df

    def random_sample_imputation(df: pd.DataFrame) -> pd.DataFrame:
        """The idea behind the random sample imputation is different from the previous ones and involves additional steps. 
        - First, it starts by creating two subsets from the original data. 
        - The first subset contains all the observations without missing data, and the second one contains those with missing data. 
        - Then, it randomly selects from each subset a random observation.
        - Furthermore, the missing data from the previously selected observation is replaced with the existing ones from the observation having all the data available.
        - Finally, the process continues until there is no more missing information.

        - Pros
            - This is an easy and straightforward technique.
            - It tackles both numerical and categorical data types.
            - There is less distortion in data variance, and it also preserves the original distribution of the data, which is not the case for mean, median, and more.
        - Cons
            - The randomness does not necessarily work for every situation, and this can infuse noise in the data, hence leading to incorrect statistical conclusions. 
            - Similarly to the mean and median, this approach also assumes that the data is missing completely at random (MCAR).


        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # Generate a copy of the dataframe
        temp_df = df.copy(deep = True)

        # To generate the list of columns with missing data.
        cols_with_missing_values = temp_df.columns[temp_df.isna().any()].tolist()

        # To run the process if there is missing data
        if len(cols_with_missing_values) > 0:
            # To loop through the columns
            for var in cols_with_missing_values:
                # extract a random sample
                random_sample_df = temp_df.loc[:,var].dropna().sample(temp_df.loc[:,var].isnull().sum())

                # re-index the randomly extracted sample
                random_sample_df.index = temp_df.loc[temp_df.loc[:,var].isnull()].index

                # replace the NA
                temp_df.loc[df.loc[:,var].isnull(), var] = random_sample_df
        
        else: # If no missing data found
            print("No missing values found in the dataframe provided.")
            
        # Return the dataframe
        return df

        

