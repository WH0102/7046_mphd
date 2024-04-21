from itertools import combinations
import pandas as pd
import numpy as np

class regression:
    def iterate_independent_variables(independent_variables: list, dependent_variables: str) -> list:
        """A simple iteraration of independent varaibles for piping of multinominal logistic regression

        Args:
            independent_variables (list): list of columns from dataframe
            dependent_variables (str): column name for dependent variables use in multinominal logistic regression

        Returns:
            list: return a variable_list and formulas
        """
        # Create empty lists
        formulas = []
        variable_list = []
        
        # Start looping the independent variables
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
                sklearn: Using sklearn.LogisticRegression(multi_class = "multinominal")
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
            return sm.MNLogit(dependent_variable, independent_variable).fit(disp = False)

        elif mode == "smf.ols": # For model using smf.ols
            # Import the important module
            import statsmodels.formula.api as smf
            # Return the model
            return smf.ols(formula = formula, data = df).fit(disp = False)
        
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
        

    def ordinal_logistic_regression(df: pd.DataFrame, mode: str = "OrderedModel", 
                                    x : list|tuple|set|str = None, y: str = None,
                                    formula: str = None) -> any:
        # Import the important packages
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        # Checking on type of independent variables if given value
        if x != None:
            if type(x) == str:
                column_name = (x)
            elif type(x) == dict | type(x) == bool:
                raise TypeError("Unsupported type!")
            else:
                column_name = tuple(x)

        # Return the model
        return OrderedModel(endog = df.loc[:,y],
                            exog = df.loc[:,column_name]).fit(disp = False)