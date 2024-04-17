import pandas as pd
# Unomment if want to display float in percentage
# pd.options.display.float_format = '{:.2%}'.format
import numpy as np
import polars as pl
from itertools import combinations
import statsmodels
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def categorical_descriptive_analysis(df: pd.DataFrame, independent_variables: list|tuple|str, dependent_variables: str, margins: bool = True) -> None:
    # To generate an internal function
    def single_convert(index_name: str, values_name:str) -> pd.DataFrame:
        # Convert with pivot table first
        temp_pt = df.pivot_table(index = index_name, values = values_name, aggfunc = len, margins = True)\
                    .rename(columns = {values_name:"count"})
        
        # Calculate percentage
        temp_pt.loc[:,"percentage"] = (temp_pt.loc[:,"count"]/ temp_pt.loc["All", "count"] * 100).round(2)

        # Perform the chi2 test
        chi2, p_value, dof, expected_freq = chi2_contingency(df.pivot_table(index = index_name, columns = values_name, aggfunc = len))

        # Return the pivot table based on margins param value
        if margins == True:
            return temp_pt, chi2, p_value, dof, expected_freq
        else: # Frop the index "ALL as margins always true upon pivoting"
            return temp_pt.drop(index = "All"), chi2, p_value, dof, expected_freq
    
    # If only 1 column need to be pirvoted
    if type(independent_variables) == str:
        # pivot the value and print the result out
        temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = independent_variables, values_name = dependent_variables)
        print(temp_pt.to_markdown(tablefmt = "pretty"))
        
        # Check for chi square
        print(f"Chi2 test between {independent_variables} and {dependent_variables} have chi2 statistics value = {chi2:,.2f} and p_value of {p_value:,.2f}")
    else: # If an iterable of column names passed to function
        try:
            # Print a statement for independent variables
            print("Descriptive Analysis for independent variables:")
            for column in independent_variables:
                print(f"Descriptive Analysis on {column}")
                # Prepare the pivot table and 
                temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = column, values_name = dependent_variables)
                # Print the table
                print(temp_pt.to_markdown(tablefmt = "pretty"))
                # Check for chi square
                print(f"Chi2 test between {column} and {dependent_variables} have chi2 statistics value = {chi2:,.2f} and p_value of {p_value:,.2f}")
                # Print a separator
                print("-----------------------------------------------------")
            
            # Print a statement for dependent variables
            print("Descriptive Analysis for dependent variables:")
            temp_pt, chi2, p_value, dof, expected_freq = single_convert(index_name = dependent_variables, values_name = independent_variables[0])
            print(temp_pt.to_markdown(tablefmt = "pretty"))
        except:
            raise ValueError(f"Please select values from dataframe columns only \n {df.columns}")

def label_encode(df: pd.DataFrame, columns: list|str|tuple, prefix:str = None, convert_numeric: bool = False) -> pd.DataFrame:
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

def iterate_independent_variables(independent_variables: list, dependent_variables: str) -> list:
    """A simple iteraration of independent varaibles for piping of multinominal logistic regression

    Args:
        independent_variables (list): list of columns from dataframe
        dependent_variables (str): column name for dependent variables use in multinominal logistic regression

    Returns:
        list: return a variable_list and formulas
    """
    formulas = []
    variable_list = []
    for num_vars in range(1, len(independent_variables) + 1):
        for combination in combinations(independent_variables, num_vars):
            # Append the independent variable into list
            variable_list.append(combination)

            # For statsmodel.formula.api
            independent_str = ' + '.join(combination)
            formula = f"{dependent_variables} ~ {independent_str}"
            formulas.append(formula)
    
    # Return both iterables
    return variable_list, formulas

def multinominal_logistic_regression(df: pd.DataFrame, mode: str = "sm.MNLogit", 
                                     x : list|tuple|set|str = None, y: str = None,
                                     formula: str = None) -> any:
    """Simple function just to produce the multinominal logistic regression from statsmodel.api

    Args:
        df (pd.DataFrame): pandas dataframe to be use for multinominal logistic regression
        mode (str): Mode of the model building.
            sm.MNLogit: Using statsmodel.api.MNLogit(y, X).fit()
            smf.ols: Using smf.ols(formula, df).fit()
        x (list | tuple | set | str): column names to be use for independent variable
        y (str): column name for dependent variable
        formula (str): Formula to be use in smf.ols(formula, df).fit()

    Raises:
        TypeError: x cannot be dictionary or boolean
        ValueError: if x or y not from pandas dataframe columns

    Returns:
        model that fit based om mode.
    """
    # Checking on type of independent variables if given value
    if x != None:
        if type(x) == str:
            column_name = (x)
        elif type(x) == dict | type(x) == bool:
            raise TypeError("Unsupported type!")
        else:
            column_name = tuple(x)

    # For model using statsmodels.api.MNLogit
    if mode == "sm.MNLogit":
        # Import the important module
        import statsmodels.api as sm
        # To fit the x and y into idnependent and dependent variable
        try:
            independent_variable = sm.add_constant(df.loc[:,column_name])
            dependent_variable = df.loc[:,y]
        except: # If the ccolumn_name and y can't located from dataframe
            raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
        
        # Fit the model and return
        return sm.MNLogit(dependent_variable, independent_variable).fit()

    elif mode == "smf.ols": # For model using smf.ols
        # Import the important module
        import statsmodels.formula.api as smf
        # Return the model
        return smf.ols(formula = formula, data = df).fit()
    
    elif mode == "sklearn": # For model using sklearn.linear_regression.LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import RepeatedStratifiedKFold

        # Use train-test split
        X_train, X_test, y_train, y_test = train_test_split(df.loc[:,column_name].values, df.loc[:,y].values, test_size=0.25)

        # define the model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        # define the model evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        # evaluate the model and collect the scores
        n_scores = cross_val_score(model, df.loc[:,column_name].values, df.loc[:,y].values, scoring='accuracy', cv=cv, n_jobs=-1)
        
        # report the model performance
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        # Return the model
        return model
    
def assess_model_fitness(model, title: str = None, yname_list: list = None) -> pd.DataFrame:
    # print the model summary
    print(model.summary(title = title, yname_list = yname_list))

    # print the odd ratio
    print("Odd Ratio =")
    print(print(np.exp(model.params).round(4).to_markdown(tablefmt = "pretty")))

    # print the likelihood ratio and its p value
    print(f"Likelihood ratio = {model.llr:,.4f} with p value of {model.llr_pvalue:,.4f}")

    # print the pseudo r value
    print(f"Pseudo R-squared value = {model.prsquared:,.4}")

    # print the aic and bic
    print(f"Akaike Information Criterion (AIC) = {model.aic:,.4f}")
    print(f"Bayesin Information Criterion (BIC) = {model.bic:,.4f}")

    # retrun a pandas dataframe
    return pd.DataFrame({"model_name":[title],
                         "likelihood_ratio":[model.llr], 
                         "llr_p_value":[model.llr_pvalue], 
                         "pseudo-r-squared":[model.prsquared], 
                         "aic_akaike_information_criterion":[model.aic], 
                         "bic_bayesin_information_criterion":[model.bic]})