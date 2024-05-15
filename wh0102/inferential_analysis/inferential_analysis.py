import numpy as np
import pandas as pd


class inferential_analysis:
    def independent_t_test_groups(df: pd.DataFrame,
                                  independent_variables : list|tuple|set|str = None, 
                                  dependent_variables: str = None) -> pd.DataFrame:
        # Import the important packages
        from scipy.stats import ttest_ind, f_oneway

        # Checking on type of independent variables if given value
        if independent_variables != None:
            if type(independent_variables) == str:
                independent_columns = (independent_variables,)
            elif type(independent_variables) == dict | type(independent_variables) == bool:
                raise TypeError("Unsupported type!")
            else:
                independent_columns = tuple(independent_variables)

        # Create an empty dataframe
        summary_df = pd.DataFrame()

        # To loop through the independent variables
        for variable in independent_variables:
            # To check for number of uniqueness for the values of independent variables
            unique_values = df.loc[:,variable].unique()

            if len(unique_values) == 2:
                t_statistic, p_value = ttest_ind(*[df.loc[df.loc[:,variable] == value, dependent_variables] for value in unique_values])
                summary_df = pd.concat([summary_df, pd.DataFrame({"variable":[variable],
                                                                  "t_statistic":[t_statistic],
                                                                  "p_values":[p_value]})], ignore_index=True)
        

        # Return the summary of t_test
        return summary_df