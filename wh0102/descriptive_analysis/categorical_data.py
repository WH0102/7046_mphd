import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

class categorical_data:
    def reverse_encode(df: pd.DataFrame, 
                       json_dict: dict[any, dict[any, any]]) -> pd.DataFrame:
        """A simple formula to change the pandas dataframe values from int to str via a json type of dictionary.

        Args:
            df (pd.DataFrame): pandas.DataFrame to be use.
            json_dict (dict[any, dict[any, any]]): The dictionary format should be in such as way that
                {column_name: {original_value : desired_value},}

        Returns:
            pd.DataFrame: Return the pandas dataframe with original value replace with desired value.
        """
        # Prepare a copy of the df to prevent inplace replacement
        temp_df = df.copy(deep = True)

        # To loop through the json_dict that passed to the function
        for key1, values1 in json_dict.items():
            # To loop through the dictionary of value to be replace for the column (key)
            for key2, values2 in values1.items():
                # Replace the value one by one
                temp_df.loc[temp_df.loc[:,key1] == key2, key1] = values2
            
            # To check for numerical
            if type(values2) == int | type(values2) == float:
                temp_df.loc[:,key1] = pd.to_numeric(temp_df.loc[:,key1])

            # To check for values that is out of range
            if len(temp_df.loc[:,key1].unique()) > len(values1.keys()):
                issue_rows = temp_df.query(f"~{key1}.isin({list(values1.values())})")
                print(f"The column of {key1} is having {len(issue_rows)} ({len(issue_rows)/len(temp_df):.2%}) rows more {len(df.loc[:,key1].unique()) - len(values1.keys())} unique value than what {values1}:")
                print(issue_rows.to_markdown(tablefmt = "pretty"))

        # Return the dataframe
        return temp_df
    
    def categorical_descriptive_analysis(df: pd.DataFrame, 
                                         independent_variables: list|tuple|str, 
                                         dependent_variables: str = None, 
                                         margins: bool = True) -> tuple[list, list]:
        """To summarize the categorical type of data in pandas.DataFrame

        Args:
            df (pd.DataFrame): Pandas DataFrame that need to be analysed
            independent_variables (list | tuple | str): Columns name from df that is independent variables, will perform chi2 test against dependent variable.
            dependent_variables (str): Columns name from df that is dependent variables. Default to None.
            margins (bool, optional): To display the margisn of pandas.pivot_table or not. Defaults to True.

        Raises:
            ValueError: If column name not within df.columns will raise this value error.

        Returns:
            chi2_positive_df, chi2_negative_df which contain column name, chi2 value and its p values will be returned
        """
        # To generate an internal function
        @staticmethod
        def single_convert(index_name: str, 
                           column_name:str) -> pd.DataFrame:
            """Internal function to pivot table a column

            Args:
                index_name (str): Column name to be pass to index in padas.pivot_table
                column_name (str): Column name to be pass to values in padas.pivot_table

            Returns:
                pd.DataFrame: Pivoted table according to index and values.
            """
            # Create a copy of df
            temp_df = df.copy(deep = True).reset_index()
            # Convert with pivot table first
            if column_name != None:
                temp_pt = temp_df.pivot_table(index = index_name, columns = column_name, values = "index", aggfunc = len, margins = True, margins_name = "All")
                
                # Calculate percentage
                temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"All"]/ temp_pt.loc["All", "All"] * 100).round(2)

                # Perform the chi2 test
                chi2, p_value, dof, expected_freq = chi2_contingency(temp_df.pivot_table(index = index_name, columns = column_name, values = "index", aggfunc = len))
            
            else:
                # Convert with  pivot table
                temp_pt = temp_df.pivot_table(index = index_name, values = "index", aggfunc = len, margins = True)\
                                 .rename(columns = {"index":"count"})
                
                # Calculate percentage
                temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"count"]/ temp_pt.loc["All", "count"] * 100).round(2)

                # just set the value for chi2, p_value, dof, expected_freq as None
                chi2, p_value, dof, expected_freq = None, None, None, None

            # Return the pivot table based on margins param value
            if margins == True:
                return temp_pt, chi2, p_value, dof, expected_freq
            else: # Frop the index "ALL as margins always true upon pivoting"
                return temp_pt.drop(index = "All"), chi2, p_value, dof, expected_freq
        
        # Prepare 2 emptly list
        chi2_positive_df = pd.DataFrame()
        chi2_negative_df = pd.DataFrame()

        # If only 1 column need to be pivoted
        if type(independent_variables) == str:
            # pivot the value and print the result out
            temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = independent_variables, column_name = dependent_variables)
            print(temp_pt.to_markdown(tablefmt = "pretty"))
            
            # Check for chi square
            if chi2 != None:
                print(f"Chi2 test between {independent_variables} and {dependent_variables} have chi2 statistics value = {chi2:,.2f} and p_value of {p_value:,.2f}")
        else: # If an iterable of column names passed to function
            try:
                # Print a statement for independent variables
                print("Descriptive Analysis for independent variables:")
                for column in independent_variables:
                    print(f"Descriptive Analysis on {column}")
                    # Prepare the pivot table and 
                    temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = column, column_name = dependent_variables)
                    
                    # Print the table
                    print(temp_pt.to_markdown(tablefmt = "pretty"))

                    # Check for chi square
                    if chi2 != None:
                        print(f"Chi2 test between {column} and {dependent_variables} have chi2 statistics value = {chi2:,.2f} and p_value of {p_value:,.2f}")
                    
                    # Print a separator
                    print("----------------------------------------------------------------")

                    # To put the column name into chi2 related list
                    if p_value != None:
                        if p_value < 0.05:
                            chi2_positive_df = pd.concat([chi2_positive_df, pd.DataFrame({"column_name":[column],
                                                                                        "chi2":[chi2],
                                                                                        "chi2_p_value":[p_value]})],
                                                                                        ignore_index = True)
                        else:
                            chi2_negative_df = pd.concat([chi2_negative_df, pd.DataFrame({"column_name":[column],
                                                                                        "chi2":[chi2],
                                                                                        "chi2_p_value":[p_value]})],
                                                                                        ignore_index = True)
                # To check for dependent variable
                if dependent_variables != None:
                    # Print a statement for dependent variables
                    print("Descriptive Analysis for dependent variables:")
                    temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = dependent_variables, column_name = None)
                    print(temp_pt.to_markdown(tablefmt = "pretty"))
            except:
                raise ValueError(f"Please select values from dataframe columns only \n {df.columns}")
            
        # Return both chi2 result list
        return chi2_positive_df, chi2_negative_df
    
    def label_encode(df: pd.DataFrame, 
                     columns: list|str|tuple, 
                     prefix:str = None, 
                     convert_numeric: bool = False) -> pd.DataFrame:
        """A function written for labelling non-numerical data in pandas Dataframe to numerical data via
        sklearn.preprocessing.LabelEncoder.

        Args:
            df (pd.DataFrame): Pandas Dataframe to be pass to the function
            columns (list | str | tuple): Columns name to label encode, raise error for dictionary or boolean type
            prefix (str): Prefix to be use during label encoding.
            convert_numeric (bool, optional): Will return columns dtype to numerical. Use with cautious as unable to re-encode once converted. Defaults to False.

        Raises:
            TypeError: Dictionary and bool are not supported

        Returns:
            pd.DataFrame: return pandas dataframe
        """
        # Import the important packages
        from sklearn.preprocessing import LabelEncoder
        import polars as pl

        # Check for the type of columns param
        if type(columns) == str:
            # Convert the column based on prefix
            if prefix == None: # If no prefix pass by user
                df.loc[:,columns] = LabelEncoder().fit_transform(df.loc[:,columns])

            elif type(prefix) == str: # If user pass a str to prefix
                df.loc[:,f"{prefix}_{columns}"] = LabelEncoder().fit_transform(df.loc[:,columns])

            else: # Raise TypeError fif user not passing str values
                raise TypeError("Support str for prefix only.")
            
        elif type(columns) == dict | type(columns) == bool: # Not compatible for pandas dataframe
            raise TypeError("Dictionary and bool are not supported")
        
        else: #columns compatible to pandas dataframe
            for column in columns:
                if prefix == None: # If no prefix pass by user
                    df.loc[:,column] = LabelEncoder().fit_transform(df.loc[:,column])

                elif type(prefix) == str: # If user pass a str to prefix
                    df.loc[:,f"{prefix}_{column}"] = LabelEncoder().fit_transform(df.loc[:,column])

                else: # Raise TypeError fif user not passing str values
                    raise TypeError("Support str for prefix only.")
        
        if convert_numeric == True: # If user wanted to convert the label encoding into integer
            return pl.from_pandas(df).to_pandas()
        else: # If user do not want to convert the label encoding into integer
            return df