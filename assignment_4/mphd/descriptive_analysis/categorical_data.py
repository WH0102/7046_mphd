import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from ..pre_processing.pre_processing import pre_processing

class categorical_data:
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
            if len(temp_df.loc[:,key1].unique()) > len(values1.keys()): # if the data unique key more than data dictionary keys
                # To prepare the dataframe rows that have issues
                issue_rows = temp_df.query(f"~{key1}.isin({list(values1.values())})")
                # Print some information and the table
                print(f"The column of {key1} is having {len(issue_rows)} ({len(issue_rows)/len(temp_df):.2%}) rows more {len(df.loc[:,key1].unique()) - len(values1.keys())} unique value than what {values1}:")
                print(issue_rows.to_markdown(tablefmt = "pretty"))

        # Return the dataframe
        return temp_df
    
    def plot_categorical_data(pivot_table:pd.DataFrame, column_name:str) -> None:
        import matplotlib.pyplot as plt
        # Plotting pie chart
        plt.pie(pivot_table, labels=pivot_table.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Pie Chart for {column_name}')
        plt.axis('equal')
        plt.show()
        plt.close()
    
    def categorical_descriptive_analysis(df: pd.DataFrame, 
                                         independent_variables: list|tuple|str, 
                                         dependent_variables: str = None, 
                                         margins: bool = True,
                                         round_value: int = 2,
                                         dependent_uniques_cut_off: int = 5,
                                         analyse_dependent:bool = False) -> tuple[list, list]:
        """To summarize the categorical type of data in pandas.DataFrame

        Args:
            df (pd.DataFrame): Pandas DataFrame that need to be analysed
            independent_variables (list | tuple | str): Columns name from df that is independent variables, will perform chi2 test against dependent variable.
            dependent_variables (str): Columns name from df that is dependent variables. Default to None.
            margins (bool, optional): To display the margisn of pandas.pivot_table or not. Defaults to True.
            round_value (int): To display the statistic table with round format or not. Defaults to 2.
            dependent_uniques_cut_off (int): Numbers used to cut of dependent variables as continous or categorical
            analyse_dependent (bool): To decide need to analyse dependent variable or not. Defauilts to False.

        Raises:
            ValueError: If column name not within df.columns will raise this value error.

        Returns:
            chi2_positive_df, chi2_negative_df which contain column name, chi2 value and its p values will be returned
        """
        import matplotlib.pyplot as plt
        # Checking on type of independent variables if given value
        independent_columns = pre_processing.identify_independent_variable(independent_variables)

        # Create a copy of df
        temp_df = df.copy(deep = True).reset_index()

        # Prepare empty statistic table
        statistic_table = pd.DataFrame()

        # Starting with independent variable analysis
        print("Descriptive Analysis for independent variables:")

        # If dependent variable not none
        if dependent_variables != None:
            # To look for either dependent variables more than cut off point or not
            if len(temp_df.loc[:,dependent_variables].unique()) >= dependent_uniques_cut_off:
                # To loop through the independent variable columns
                for variable in independent_columns:
                    # Convert with  pivot table
                    temp_pt = temp_df.pivot_table(index = variable, values = "index", 
                                                  aggfunc = len, margins = True)\
                                    .rename(columns = {"index":"count"})
                    
                    # Calculate percentage
                    temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"count"]/ temp_pt.loc["All", "count"] * 100).round(2)

                    # Check for margins
                    if margins == False:
                        temp_pt = temp_pt.drop(index = "All")

                    # print the dataframe
                    print(temp_pt.to_markdown(tablefmt = "pretty"))

                    if margins == True:
                        temp_pt = temp_pt.drop(index = "All")

                    # To plot pie/bar chart for independent_columns variable vs continous dependent data
                    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
                    ax1.pie(temp_pt.loc[:,"count"], labels = temp_pt.index, autopct = '%1.2f%%')
                    ax1.set_title(f"Pie chart for {variable}")
                    plt.show()

                    # To check for unique values in the independent variable
                    unique_values = temp_df.loc[:,variable].unique()
                    if len(unique_values) == 2:
                        # Perform the independent t test
                        t_statistic, p_value = ttest_ind(*[temp_df.loc[temp_df.loc[:,variable] == value, dependent_variables] for value in unique_values])
                        
                        # print the independent t test information
                        print(f"Independent t test between {variable} and {dependent_variables} have t statistics value = {t_statistic:,.2f} and p_value of {p_value:,.2f}")

                        # Concat the statistic table
                        statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                    "test_name":["Independent_t_test"],
                                                                                    "statistic_values":[t_statistic],
                                                                                    "p_values":[p_value]})], ignore_index=True)
                        
                    elif len(unique_values) > 2:
                        # Perform one way anova test
                        f_statistic, p_vlue = f_oneway(*[temp_df.loc[temp_df.loc[:,variable] == value, dependent_variables] for value in unique_values])

                        # print the one way ANOVA information
                        print(f"One Way ANOVA between {variable} and {dependent_variables} have F statistics value = {f_statistic:,.2f} and p_value of {p_value:,.2f}")

                        # Concat the statistic table
                        statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                    "test_name":["One_Way_ANOVA"],
                                                                                    "statistic_values":[f_statistic],
                                                                                    "p_values":[p_value]})], ignore_index=True)

                    # Print a separator
                    print("----------------------------------------------------------------")

            else: # len(temp_df.loc[:,dependent_variables].unique()) < dependent_uniques_cut_off
                for variable in independent_columns:
                    temp_pt = temp_df.pivot_table(index = variable, columns = dependent_variables, values = "index", 
                                                  aggfunc = len, margins = True, margins_name = "All")
                    
                    # Calculate percentage
                    temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"All"]/ temp_pt.loc["All", "All"] * 100).round(2)

                    # Check for margins
                    if margins == False:
                        temp_pt = temp_pt.drop(index = "All")

                    #  print the dataframe
                    print(temp_pt.to_markdown(tablefmt = "pretty"))

                    if margins == True:
                        temp_pt = temp_pt.drop(index = "All")

                    # To plot pie/bar chart for independent_columns variable vs categorical dependent data
                    fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
                    ax1.pie(temp_pt.loc[:,"All"], labels = temp_pt.index, autopct = '%1.2f%%')
                    ax1.set_title(f"Pie chart for {variable}")
                    plt.show()

                    # Perform the chi2 test
                    chi2, p_value, dof, expected_freq = chi2_contingency(temp_df.pivot_table(index = variable, columns = dependent_variables, 
                                                                                            values = "index", aggfunc = len))
                    
                    # print the chi2 information
                    print(f"Chi2 test between {variable} and {dependent_variables} have chi2 statistics value = {chi2:,.2f} and p_value of {p_value:,.2f}")

                    # Print a separator
                    print("----------------------------------------------------------------")

                    # Concat the statistic table
                    statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                "test_name":["Chi2 Test"],
                                                                                "statistic_values":[chi2],
                                                                                "p_values":[p_value]})], ignore_index=True)
                    
                # Print a statement for dependent variables
                print("Descriptive Analysis for dependent variables:")
                # Prepare the variables based on the single convert function
                temp_pt = temp_df.pivot_table(index = dependent_variables, values = "index", 
                                                aggfunc = len, margins = True)\
                                    .rename(columns = {"index":"count"})
                
                if margins == False:
                    temp_pt = temp_pt.drop(index = "All")

                # print the pivot table
                print(temp_pt.to_markdown(tablefmt = "pretty"))

                if margins == True:
                    temp_pt = temp_pt.drop(index = "All")

                # To plot pie/bar chart for dependent_columns variable if is categorical data only
                fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
                ax1.pie(temp_pt.loc[:,"count"], labels = temp_pt.index, autopct = '%1.2f%%')
                ax1.set_title(f"Pie chart for {dependent_variables}")
                plt.show()

            # Print the statistic_table based on round value
            print("The summary of the statistical tests for independent variable:")
            if round_value != None:
                print(statistic_table.round(round_value).to_markdown(tablefmt = "pretty"))
            else:
                print(statistic_table.to_markdown(tablefmt = "pretty"))

            # Collect for positive  and negative statistical test
            positive_p_value_columns = [row["independent_variable"] for index, row in statistic_table.iterrows() if row["p_values"] < 0.05]
            negative_p_value_columns = [row["independent_variable"] for index, row in statistic_table.iterrows() if row["p_values"] >= 0.05]

            # To print the intepretation for positive chi2 square test
            if len(positive_p_value_columns) > 0:
                print(f"""The chi2 test is positive and indicate there is an association between {dependent_variables} and {", ".join(positive_p_value_columns)}.""")
            
            # To print the intepretation for negative chi2 square test
            if len(negative_p_value_columns) > 0:
                print(f"""The chi2 test is negative and indicate there is no association between {dependent_variables} and {", ".join(negative_p_value_columns)}""")
        
        else: # dependent_variables == None:
            # To loop through the independent variable columns
            for variable in independent_columns:
                # Convert with  pivot table
                temp_pt = temp_df.pivot_table(index = variable, values = "index", 
                                            aggfunc = len, margins = True)\
                                .rename(columns = {"index":"count"})
                
                # Calculate percentage
                temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"count"]/ temp_pt.loc["All", "count"] * 100).round(2)

                # Check for margins
                if margins == False:
                    temp_pt = temp_pt.drop(index = "All")

                # print the dataframe
                print(temp_pt.to_markdown(tablefmt = "pretty"))

                if margins == True:
                    temp_pt = temp_pt.drop(index = "All")

                # To plot pie/bar chart for independent_columns variable vs continous dependent data
                fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
                ax1.pie(temp_pt.loc[:,"count"], labels = temp_pt.index, autopct = '%1.1f%%')
                ax1.set_title(f"Pie chart for {variable}")
                plt.show()

                # Print a separator
                print("----------------------------------------------------------------")

        # To analyse dependent variables
        if analyse_dependent == True:
            # Convert with  pivot table
            temp_pt = temp_df.pivot_table(index = dependent_variables, values = "index", 
                                          aggfunc = len, margins = True)\
                            .rename(columns = {"index":"count"})
            
            # Calculate percentage
            temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"count"]/ temp_pt.loc["All", "count"] * 100).round(2)

            print("-----------------------------------------------------------")
            print("For dependent variable:")

            # Check for margins
            if margins == False:
                temp_pt = temp_pt.drop(index = "All")

            # print the dataframe
            print(temp_pt.to_markdown(tablefmt = "pretty"))

            if margins == True:
                temp_pt = temp_pt.drop(index = "All")

            # To plot pie/bar chart for independent_columns variable vs continous dependent data
            fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
            ax1.pie(temp_pt.loc[:,"count"], labels = temp_pt.index, autopct = '%1.2f%%')
            ax1.set_title(f"Pie chart for {variable}")
            plt.show()

        # Return empty statistic table
        return statistic_table
    
    def descriptive_distribution_analysis(df:pd.DataFrame, 
                                          independent_variables: list|tuple|str, 
                                          dependent_variables:str = None,
                                          data_dictionary:dict = None,
                                          statistical_test:str|None = None,
                                          round_value: int|None = 2):
        # Import the important packages
        import matplotlib.pyplot as plt

        # Checking on type of independent variables if given value
        independent_variables = pre_processing.identify_independent_variable(independent_variables)

        # Ensure the independent variables are all categorical

        # Create a copy of the df
        temp_df = df.loc[:,independent_variables + (dependent_variables,)].copy(deep = True).reset_index()

        # Final table
        summary_table = []
        statistic_table = pd.DataFrame()

        # To loop if more than 1 columns provided
        for variable in independent_variables:
            temp_pt = temp_df.pivot_table(index = variable, values = dependent_variables, aggfunc=[np.mean, np.std, np.var, len])
            # Rename the column name
            temp_pt.columns = [name[0] for name in temp_pt.columns]
            # Calculate the percentage of changes between var and mean by using mean as denominator
            temp_pt.loc[:,("dispersion_ratio")] = temp_pt.loc[:,("var")] / temp_pt.loc[:,("mean")]
            # Calculate len_percentage
            temp_pt.loc[:,("len_percentage")] = temp_pt.loc[:,("len")] / len(temp_df)

            # Print based on round_value value
            if round_value == None:
                print(temp_pt.to_markdown(tablefmt = "pretty"))
            else:
                print(temp_pt.round(round_value).to_markdown(tablefmt = "pretty"))

            # Test on uniques values for independent variables
            unique_values = temp_df.loc[:,variable].unique()
            # Statistical Test
            if statistical_test == "t_test":
                # Perform t test if independent variables only have 2 distinct values
                if len(unique_values) == 2:
                    # Perform the independent t test
                    t_statistic, p_value = ttest_ind(*[temp_df.loc[temp_df.loc[:,variable] == value, dependent_variables] for value in unique_values])
                    
                    # print the independent t test information
                    print(f"Independent t test between {variable} and {dependent_variables} have t statistics value = {t_statistic:,.2f} and p_value of {p_value:,.2f}")

                    # Concat the statistic table
                    statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                "test_name":["Independent_t_test"],
                                                                                "statistic_values":[t_statistic],
                                                                                "p_values":[p_value]})], ignore_index=True)
                    
                elif len(unique_values) > 2:
                    # Perform one way anova test
                    f_statistic, p_vlue = f_oneway(*[temp_df.loc[temp_df.loc[:,variable] == value, dependent_variables] for value in unique_values])

                    # print the one way ANOVA information
                    print(f"One Way ANOVA between {variable} and {dependent_variables} have F statistics value = {f_statistic:,.2f} and p_value of {p_value:,.2f}")

                    # Concat the statistic table
                    statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                "test_name":["One_Way_ANOVA"],
                                                                                "statistic_values":[f_statistic],
                                                                                "p_values":[p_value]})], ignore_index=True)
                    
            elif statistical_test == "point_biserial_correlations":
                # import the necessary modeul
                from scipy.stats import pointbiserialr
                # Perform t test if independent variables only have 2 distinct values
                if len(unique_values) == 2:
                    # Perform the independent t test
                    point_biserial_corr, p_value = pointbiserialr(temp_df.loc[:,variable], temp_df.loc[:,dependent_variables])
                    
                    # print the independent t test information
                    if p_value < 0.05:
                        print(f"There is a statistically significant correlation between {variable} and {dependent_variables} as the p value of {p_value:,.2f} less than 0.05 and the point biserial correlation = {point_biserial_corr:,.2f}.")
                    else:
                        print(f"There is no statistically significant correlation between {variable} and {dependent_variables} as the p value of {p_value:,.2f} more than 0.05 and the point biserial correlation = {point_biserial_corr:,.2f}.")

                    # Concat the statistic table
                    statistic_table = pd.concat([statistic_table, pd.DataFrame({"independent_variable":[variable],
                                                                                "test_name":["Point Biserial Correlations"],
                                                                                "statistic_values":[point_biserial_corr],
                                                                                "p_values":[p_value]})], ignore_index=True)
                    
            # To rename on the temp_pt
            if data_dictionary != None:
                temp_pt.index = temp_pt.index.map(data_dictionary[variable])

            # Append to the list
            summary_table.append(temp_pt)

            # To plot historgram
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
            ax1.pie(temp_pt.loc[:,"len"], labels = temp_pt.index, autopct = '%1.2f%%')
            ax1.set_title(f"Pie chart for {variable}")

            # To plot pie/bar chart for independent_columns variable vs continous dependent data
            for value in df.loc[:,variable].unique():
                ax2.hist(df.loc[df.loc[:,variable] == value, dependent_variables],  
                         alpha=0.5, # the transaparency parameter 
                         label=value)

            plt.legend(loc='upper right') 
            plt.title(f'Distribution of {variable} against {dependent_variables}') 
            plt.show()
            plt.close()

        # Display the statistic table
        print(statistic_table.round(round_value).to_markdown(tablefmt = "pretty", index = False))
        # Interpret the statistic table
        # Collect for positive  and negative statistical test
        positive_p_value_columns = [row["independent_variable"] for index, row in statistic_table.iterrows() if row["p_values"] < 0.05]
        negative_p_value_columns = [row["independent_variable"] for index, row in statistic_table.iterrows() if row["p_values"] >= 0.05]

        # To print the intepretation for positive chi2 square test
        if len(positive_p_value_columns) > 0:
            print(f"""Theere is an association between {dependent_variables} and {", ".join(positive_p_value_columns)} as their {statistic_table["test_name"][0]} is positive with p value less than 0.05""")
        
        # To print the intepretation for negative chi2 square test
        if len(negative_p_value_columns) > 0:
            print(f"""There is no association between {dependent_variables} and {", ".join(negative_p_value_columns)} as their {statistic_table["test_name"][0]} is negative with p value more than 0.05""")

        # Generate depedent variables
        dependent_mean = np.mean(df.loc[:,dependent_variables])
        dependent_std = np.std(df.loc[:,dependent_variables])
        dependent_var = np.var(df.loc[:,dependent_variables])
        dependent_len = len(df.loc[:,dependent_variables])
        dependent_percentage = dependent_var / dependent_mean
        dependent_description = pd.DataFrame({"index":[dependent_variables],
                                              "mean":[dependent_mean], 
                                              "std":[dependent_std], 
                                              "var":[dependent_var], 
                                              "len":[dependent_len], 
                                              "dispersion_ratio":[dependent_percentage],
                                              "len_percentage":[1]}).set_index("index")
        
        # Generate summary table
        summary_table = pd.concat(summary_table + [dependent_description,])
        
        # To combine all the table and return
        return summary_table, statistic_table