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
    
    def regression(df: pd.DataFrame, mode: str = "sm.MNLogit", 
                   independent_variables : list|tuple|set|str = None, 
                   dependent_variables: str = None,
                   formula: str = None) -> any:
        """Simple function just to produce the regression from statsmodel.api

        Args:
            df (pd.DataFrame): pandas dataframe to be use for regression
            mode (str): Mode of the model building.
                sm.MNLogit: Using statsmodel.api.MNLogit(y, X).fit()
                smf.ols: Using smf.ols(formula, df).fit()
                sklearn: Using sklearn.LogisticRegression(multi_class = "multinominal")
                OrderedModel: Using statsmodels.miscmodels.ordinal_model.OrderedModel().fit()
            independent_variables (list | tuple | set | str): column names to be use for independent variable
            dependent_variables (str): column name for dependent variable
            formula (str): Formula to be use in smf.ols(formula, df).fit()

        Raises:
            TypeError: x cannot be dictionary or boolean
            ValueError: if x or y not from pandas dataframe columns

        Returns:
            model that fit based om mode.
        """
        # Checking on type of independent variables if given value
        if independent_variables != None:
            if type(independent_variables) == str:
                column_name = (independent_variables)
            elif type(independent_variables) == dict | type(independent_variables) == bool:
                raise TypeError("Unsupported type!")
            else:
                column_name = tuple(independent_variables)

        # For model using statsmodels.api.MNLogit
        if mode == "sm.MNLogit":
            # Import the important module
            import statsmodels.api as sm
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                dependent_variable = df.loc[:,dependent_variables]
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
            X_train, X_test, y_train, y_test = train_test_split(df.loc[:,column_name].values, df.loc[:,dependent_variables].values, test_size=0.25)

            # define the model
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

            # define the model evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

            # evaluate the model and collect the scores
            n_scores = cross_val_score(model, df.loc[:,column_name].values, df.loc[:,dependent_variables].values, scoring='accuracy', cv=cv, n_jobs=-1)
            
            # report the model performance
            print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

            # Return the model
            return model
        
        elif mode == "OrderedModel":
            # Import the important packages
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            # Return the model
            return OrderedModel(endog = df.loc[:,dependent_variables],
                                exog = df.loc[:,column_name]).fit(disp = False)
        
    def regression_list(df: pd.DataFrame, mode: str = "sm.MNLogit", 
                        independent_variables : list|tuple|set|str = None, 
                        dependent_variables: str = None):
        variables_list, formula_list = regression.iterate_independent_variables(independent_variables = independent_variables,
                                                                                dependent_variables = dependent_variables)
        
        # To generate list of model based on regression mode
        if mode != "smf.ols":
            model_list = [regression.regression(df = df, mode = mode, independent_variables = independent_variable, dependent_variables = dependent_variables)
                        for independent_variable in variables_list]
        else:
            model_list = [regression.regression(df = df, mode = mode, formula = formula) for formula in formula_list]

        # To generate summary of model in pandas.DataFrame
        if mode == "sm.MNLogit":
            summary_model = pd.DataFrame({"model":model_list,
                                          "formula":formula_list,
                                          "variables":[",".join(variable) for variable in variables_list],
                                          "pseudo-r-2":[model.prsquared for model in model_list],
                                          "log-likelihood":[model.llf for model in model_list],
                                          "llr_p_value":[model.llr_pvalue for model in model_list],
                                          "aic_akaike_information_criterion":[model.aic for model in model_list],
                                          "bic_bayesin_information_criterion":[model.bic for model in model_list]})
            
        elif mode == "OrderedModel":
            summary_model = pd.DataFrame({"model":model_list,
                                          "formula":formula_list,
                                          "variables":[",".join(variable) for variable in variables_list],
                                          "pseudo-r-2":[model.prsquared for model in model_list],
                                          "log-likelihood":[model.llf for model in model_list],
                                          "llr_p_value":[model.llr_pvalue for model in model_list],
                                          "aic_akaike_information_criterion":[model.aic for model in model_list],
                                          "bic_bayesin_information_criterion":[model.bic for model in model_list]})

        # To loop through the model list with len of model list
        for num in range(0, len(model_list)):
            # Print the model summary
            print(model_list[num].summary(title = formula_list[num]))

            # Print separator
            print("----------------------------------------------------------------------------------------------------")

        # Return the necessary list
        return model_list, summary_model