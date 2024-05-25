import pandas as pd
import numpy as np
# from ..missing_values.missing_values import missing_values

class pre_processing:
    def identify_independent_variable(independent_variables : list|tuple|set|str = None) -> tuple:
        if independent_variables != None:
            if type(independent_variables) == str:
                return (independent_variables,)
            elif type(independent_variables) == dict | type(independent_variables) == bool:
                raise TypeError("Unsupported type!")
            else:
                return tuple(independent_variables)
            
    def check_duplication(df: pd.DataFrame, 
                          duplication_subset: list|str|None = None,
                          round_value:int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        # To prepare the duplicated from all
        if duplication_subset == None:
            duplicated_df = df.loc[df.duplicated(keep = False)]
            to_drop_duplicated_df = df.loc[df.duplicated()]
        else: # If user provide duplication_subset
            # To tuple the duplication_subset prior to form the dataframe
            duplication_subset = list(pre_processing.identify_independent_variable(duplication_subset))
            duplicated_df = df.loc[df.duplicated(subset = duplication_subset, keep = False)]
            to_drop_duplicated_df = df.loc[df.druplicated(subset = duplication_subset)]

        # Print a statement
        if len(duplicated_df) > 0: # Duplication found
            print(f"""A total of {len(duplicated_df)} ({round(len(duplicated_df)/len(df)*100, round_value)}%) detected in the dataframe. \
From which a total of {len(to_drop_duplicated_df)} ({round(len(to_drop_duplicated_df)/len(df)*100, round_value)}%) can be drop out from the dataframe.""")
            return duplicated_df, to_drop_duplicated_df
        else: # No duplication found
            print("No duplication found in the dataset.")
            return pd.DataFrame(), pd.DataFrame()
        
    def train_test_split(df:pd.DataFrame, 
                         independent_variables : list|tuple|set|str,
                         dependent_variable:str,
                         test_size:float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # import the necessary packages
        from sklearn.model_selection import train_test_split

        # Prepare the independent variables
        independent_variables = pre_processing.identify_independent_variable(independent_variables)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:,independent_variables],
                                                            df.loc[:,dependent_variable],
                                                            test_size=test_size,
                                                            stratify=df.loc[:,dependent_variable])
        
        # Return all dataframe
        return X_train, X_test, y_train, y_test

            
        
