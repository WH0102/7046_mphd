import pandas as pd
import numpy as np
from ..pre_processing.pre_processing import pre_processing
import statsmodels.api as sm

class regression:
    def iterate_independent_variables(independent_variables: list, dependent_variables: str) -> list:
        """A simple iteraration of independent varaibles for piping of multinominal logistic regression

        Args:
            independent_variables (list): list of columns from dataframe
            dependent_variables (str): column name for dependent variables use in multinominal logistic regression

        Returns:
            list: return a variable_list and formulas
        """
        from itertools import combinations
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
                sm.Logit: Using statsmodel.api.Logit(y, X).fit()
                smf.ols: Using smf.ols(formula, df).fit()
                smf.logit: Using smf.logit(formula, df).fit()
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
        column_name = pre_processing.identify_independent_variable(independent_variables)

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
        
        elif mode == "sm.Logit":
            import statsmodels.api as sm
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                dependent_variable = df.loc[:,dependent_variables]
            except: # If the ccolumn_name and y can't located from dataframe
                raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
            
            # Fit the model and return
            return sm.Logit(dependent_variable, independent_variable).fit(disp = False)

        elif mode == "smf.ols": # For model using smf.ols
            # Import the important module
            import statsmodels.formula.api as smf
            # Return the model
            return smf.ols(formula = formula, data = df).fit(disp = False)

        elif mode == "smf.logit": # For model using smf.ols
            # Import the important module
            import statsmodels.formula.api as smf
            # Return the model
            return smf.logit(formula = formula, data = df).fit(disp = False)
        
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
        
        elif mode == "poisson_regression":
            # Import the important packages
            import statsmodels.api as sm
            # from statsmodels.discrete.discrete_model import Poisson
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                # independent_variable = df.loc[:,column_name]
                dependent_variable = df.loc[:,dependent_variables]
            except: # If the ccolumn_name and y can't located from dataframe
                raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
            
            # # Fit the model and return
            # return sm.GLM(dependent_variable, independent_variable, family=sm.families.Poisson()).fit(disp = False)

            # Fit the model and return
            return sm.Poisson(dependent_variable, independent_variable).fit(disp = False)

        elif mode == "negative_binominal_regression":
            # Import the important package
            import statsmodels.api as sm
            # from statsmodels.discrete.discrete_model import NegativeBinomial
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                # independent_variable = df.loc[:,column_name]
                dependent_variable = df.loc[:,dependent_variables]
            except: # If the ccolumn_name and y can't located from dataframe
                raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
            
            # Fit the model and return
            return sm.NegativeBinomial(dependent_variable, independent_variable).fit(disp = False)
        
        elif mode == "zero_inflatted_poisson_regression":
            # import the necessary packages
            import statsmodels.api as sm
            # from statsmodels.discrete.count_model import ZeroInflatedPoisson
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                # independent_variable = df.loc[:,column_name]
                dependent_variable = df.loc[:,dependent_variables]
            except: # If the ccolumn_name and y can't located from dataframe
                raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
            
            # Fit the model and return
            return sm.ZeroInflatedPoisson(endog = dependent_variable, 
                                          exog = independent_variable, 
                                          exog_infl = independent_variable, 
                                          inflation='logit').fit(disp = False)
            # return sm.ZeroInflatedPoisson(dependent_variable, independent_variable).fit(disp = False)
        
        elif mode == "zero_inflatted_negative_binominal_regression":
            # Import important packages
            import statsmodels.api as sm
            # from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
            # To fit the x and y into idnependent and dependent variable
            try:
                independent_variable = sm.add_constant(df.loc[:,column_name])
                # independent_variable = df.loc[:,column_name]
                dependent_variable = df.loc[:,dependent_variables]
            except: # If the ccolumn_name and y can't located from dataframe
                raise ValueError(f"Please select columns name from your dataframe correctly. \n{df.columns}")
            
            # Fit the model and return
            return sm.ZeroInflatedNegativeBinomialP(endog = dependent_variable, 
                                                    exog = independent_variable, 
                                                    inflation='probit').fit(disp = False)
            # return sm.ZeroInflatedNegativeBinomialP(dependent_variable, independent_variable).fit(disp = False)

    def generate_params_df(model:any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """To generate the params and exponential params of coefficients for statsmodel regression.

        Args:
            model (any): For putting model generated from statsmodel only

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The first table is the params of coefficient while the second dataframe is the exponential for the first dataframe.
        """
        # Generate params table
        params = pd.DataFrame(model.params, columns = ["coefficient"])
        exp_params = np.exp(params)
        bse = pd.DataFrame(model.bse, columns = ["std_err"])
        exp_bse = np.exp(bse)
        t_statistic = pd.DataFrame(model.tvalues, columns = ["t_statistic"])
        p_value = pd.DataFrame(model.pvalues, columns = ["p_value"])
        conf_interval = model.conf_int().rename(columns = {0:"2.5%", 1:"97.5%"})
        exp_conf_interval = np.exp(conf_interval)
        # Concat in axis 1
        params_df = pd.concat([params, bse, t_statistic, p_value, conf_interval], axis= 1)
        # Exponential the params for
        exp_params_df = pd.concat([exp_params, exp_bse, t_statistic, p_value, exp_conf_interval], axis= 1)

        # Return both params_df and exp_params_df
        return params_df, exp_params_df

    def generate_wald_test(model:any) -> pd.DataFrame:
        """Based on the statistical model to generate the wald test

        Args:
            model (any): Statsmodel.api regression model

        Returns:
            pd.DataFrame: Consist of params as indes, chi2 value, p value for the chi2 and degree of freedom. 
        """
        # To get the wald test in pd.DataFrame
        wald_test_df = model.wald_test_terms().summary_frame()
        # To convert the information in wald test for further usage
        wald_test_df['chi2'] = wald_test_df['chi2'].apply(lambda x: x[0][0])
        wald_test_df["P>chi2"] = wald_test_df["P>chi2"].astype(float)
        # Return the df
        return wald_test_df

    def regression_list(df: pd.DataFrame, mode: str = "sm.MNLogit", 
                        independent_variables : list|tuple|set|str = None, 
                        dependent_variables: str = None,
                        p_value_cut_off:float = 0.025) -> tuple[list, pd.DataFrame]:
        """To generate multiple statsmodel regression based on possible combination of the independent variables

        Args:
            df (pd.DataFrame): pandas dataframe to be use for regression
            mode (str): Mode of the model building.
                sm.MNLogit: Using statsmodel.api.MNLogit(y, X).fit()
                sm.Logit: Using statsmodel.api.Logit(y, X).fit()
                smf.ols: Using smf.ols(formula, df).fit()
                smf.logit: Using smf.logit(formula, df).fit()
                sklearn: Using sklearn.LogisticRegression(multi_class = "multinominal")
                OrderedModel: Using statsmodels.miscmodels.ordinal_model.OrderedModel().fit()
            independent_variables (list | tuple | set | str): column names to be use for independent variable
            dependent_variables (str): column name for dependent variable
            formula (str): Formula to be use in smf.ols(formula, df).fit()
            p_value_cut_off (float): alpha value for the coefficient p value cut off point.

        Returns:
            tuple[list, pd.DataFrame]: the list contain all the possible combination of the regression based on independent variables while the dataframe consist of possible summary for the list of regression models.
        """
        import statsmodels.api as sm
        variables_list, formula_list = regression.iterate_independent_variables(independent_variables = independent_variables,
                                                                                dependent_variables = dependent_variables)
        
        # To generate list of model based on regression mode
        if mode != "smf.ols":
            model_list = [regression.regression(df = df, mode = mode, independent_variables = independent_variable, dependent_variables = dependent_variables)
                        for independent_variable in variables_list]
        else:
            model_list = [regression.regression(df = df, mode = mode, formula = formula) for formula in formula_list]

        # To generate summary of model in pandas.DataFrame
        summary_model = pd.DataFrame()
        for num in range(0, len(model_list)):
            # Locate model
            model = model_list[num]

            # Generate params table
            params_df, exp_params_df = regression.generate_params_df(model)

            if mode == "sm.MNLogit" or mode == "OrderedModel":
                temp_summary = pd.DataFrame({"model":[model],
                                             "formula":[formula_list[num]],
                                             "variables":[",".join(variables_list[num])],
                                             "pseudo_r_2":[model.prsquared],
                                             "log_likelihood":[model.llf],
                                             "llr_p_value":[model.llr_pvalue],
                                             "aic_akaike_information_criterion":[model.aic],
                                             "bic_bayesin_information_criterion":[model.bic],
                                             "coeff_all_significant":[False if len(params_df.loc[params_df.loc[:,"p_value"] >= p_value_cut_off]) > 0 else True]})
                
                # Generate params table information into summary table
                # for column in ["coefficient", "p_value"]:
                #     for index, row in params_df.iterrows():
                #         temp_summary.loc[:,f"{column}_{index}"] = row[column]

                # # Combine the temp_summary with summary_model
                # summary_model = pd.concat([summary_model, temp_summary], ignore_index=True)
                
                
            elif mode == "sm.Logit":
                from sklearn.metrics import roc_curve, auc
                from scipy.stats import shapiro
                from statsmodels.stats.diagnostic import het_breuschpagan

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(df.loc[:,dependent_variables], model.predict(sm.add_constant(df.loc[:,variables_list[num]])))
                # Check for het_beruschpagan
                lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(model.resid_response, model.model.exog)

                # Generate a summary table
                temp_summary = pd.DataFrame({"model":[model],
                                             "formula":[formula_list[num]],
                                             "variables":[",".join(variables_list[num])],
                                             "num_variables":[len(variables_list[num])],
                                             "pseudo_r_2":[model.prsquared],
                                             "log_likelihood":[model.llf],
                                             "llr_p_value":[model.llr_pvalue],
                                             "aic_akaike_information_criterion":[model.aic],
                                             "bic_bayesin_information_criterion":[model.bic],
                                             "coeff_all_significant":[False if len(params_df.loc[params_df.loc[:,"p_value"] >= p_value_cut_off]) > 0 else True],
                                             "roc":[auc(fpr, tpr)],
                                             "shapiro_residual":[shapiro(model.resid_response)[1]],
                                             "Lagrange_Multiplier":[lm],
                                             "Lagrange_Multiplier_p-value":[lm_p_value],
                                             "F-statistic":[fvalue],
                                             "F-statistic_p-value":[f_p_value]})
                
                # # Generate params table information into summary table
                # for column in ["coefficient", "p_value"]:
                #     for index, row in params_df.iterrows():
                #         temp_summary.loc[:,f"{column}_{index}"] = row[column]

                # # Combine the temp_summary with summary_model
                # summary_model = pd.concat([summary_model, temp_summary], ignore_index=True)
                                                 
            
            elif mode == "smf.ols" or mode == "smf.logit":
                summary_model = pd.DataFrame({"model":model_list,
                                            "formula":formula_list,
                                            "variables":[",".join(variable) for variable in variables_list],
                                            "r_squared":[model.rsquared for model in model_list],
                                            "adjusted_r_squared":[model.rsquared_adj for model in model_list],
                                            "f_statistic":[model.fvalue for model in model_list],
                                            "f_statistic_p_value":[model.f_pvalue for model in model_list],
                                            "Omnibus":[float(str.strip(model.summary().tables[2].data[0][1])) for model in model_list],
                                            "Omnibus_p_value":[float(str.strip(model.summary().tables[2].data[1][1])) for model in model_list],
                                            "Jarque-Bera":[float(str.strip(model.summary().tables[2].data[1][3])) for model in model_list],
                                            "Jarque-Bera_p_value":[float(str.strip(model.summary().tables[2].data[2][3])) for model in model_list],
                                            "Durbin-Watsom":[float(str.strip(model.summary().tables[2].data[0][3])) for model in model_list],
                                            "aic_akaike_information_criterion":[model.aic for model in model_list],
                                            "bic_bayesin_information_criterion":[model.bic for model in model_list]})
                
            # elif mode == "poisson_regression" or mode == "negative_binominal_regression":
                                                            #   "pearson_chi2":[model.pearson_chi2],
                                            #                  "deviance":[model.deviance],
                                            #   "residual_deviance_df":[np.sum(np.abs(model.resid_deviance))/model.df_resid],
                                            #   "deviance_goodness_of_fit_test":[model.deviance/model.df_resid],
                                            #   "log_likelihood":[model.llf],
                # temp_summary = pd.DataFrame({"model":[model],
                #                               "formula":[formula_list[num]],
                #                               "variables":[",".join(variables_list[num])],
                #                               "num_variables":[len(variables_list[num])],
                #                               "pseudo_r_2":[model.prsquared],
                #                               "dispersion_stats":[(model.resid_pearson**2).sum() / model.df_resid],
                #                               "model_mean":[np.mean(model.fittedvalues)],
                #                               "model_var":[np.var(model.fittedvalues)],
                #                               "aic_akaike_information_criterion":[model.aic],
                #                               "bic_bayesin_information_criterion":[model.bic],
                #                               "coeff_all_significant":[False if len(params_df.loc[params_df.loc[:,"p_value"] >= p_value_cut_off]) > 0 else True],
                #                               })
                # # Generate params table information into summary table
                # for column in ["coefficient", "p_value"]:
                #     for index, row in params_df.iterrows():
                #         temp_summary.loc[:,f"{column}_{index}"] = row[column]

                # # Combine the temp_summary with summary_model
                # summary_model = pd.concat([summary_model, temp_summary], ignore_index=True)

            elif mode == "poisson_regression" or mode == "negative_binominal_regression" or mode == "zero_inflatted_poisson_regression" or mode == "zero_inflatted_negative_binominal_regression":
                                            #   "model_mean":[model.get_distribution().mean().mean()],
                                            #   "model_var":[model.get_distribution().var().mean()],
                temp_summary = pd.DataFrame({"model":[model],
                                              "formula":[formula_list[num]],
                                              "variables":[",".join(variables_list[num])],
                                              "num_variables":[len(variables_list[num])],
                                              "converged":[model.converged],
                                              "pseudo_r_2":[model.prsquared],
                                              "dispersion_stats":[(model.resid_pearson**2).sum() / model.df_resid],
                                              "log_likelihood":[model.llf],
                                              "ll_null":[model.llnull],
                                              "llr_p_value":[model.llr_pvalue],
                                              "aic_akaike_information_criterion":[model.aic],
                                              "bic_bayesin_information_criterion":[model.bic],
                                              "coeff_all_significant":[False if len(params_df.loc[params_df.loc[:,"p_value"] >= p_value_cut_off]) > 0 else True],
                                              })

            # Generate params table information into summary table
            for column in ["coefficient", "p_value"]:
                for index, row in params_df.iterrows():
                    temp_summary.loc[:,f"{column}_{index}"] = row[column]

            # Combine the temp_summary with summary_model
            summary_model = pd.concat([summary_model, temp_summary], ignore_index=True)

        # Return the necessary list
        return model_list, summary_model
    
    def model_formula(model, mode:str = None, round_value: int = 4, **kwargs) -> float:
        # Generate the params table
        params_df, exp_params_df = regression.generate_params_df(model)
        # Generate a temporary list for coefficients
        coefficient_list = set([column for column in params_df.index if column != 'const'])
        # State teh constant value in none params first
        const = params_df.loc["const", "coefficient"]
        
        # To check on kwargs value first
        if set(coefficient_list) != set(kwargs.keys()):
            raise ValueError(f"The coefficient required are {coefficient_list}")

        # To prepare the formula based on mode
        for key, value in kwargs.items():
            const += value * params_df.loc[value, "coefficient"]
        
        # Testing
        return const, np.exp(const)
        # To generate partial formula
        # coefficient_list = list(zip(coefficient_list, 
        #                             params_df.loc[params_df.index != "const","coefficient"], 
        #                             exp_params_df.loc[exp_params_df.index != "const","coefficient"]))
        # params_partial_formula = " + ".join([f"({round(value[1], round_value)} * {value[0]})" for value in coefficient_list])
        # params_partial_formula = f"{params_df.loc["const", "coefficient"]} + {params_partial_formula}"
        # exp_partial_formula = " + ".join([f"({value[2]:.2f} * {value[0]})" for value in coefficient_list])
        # exp_partial_formula = f"{round(np.exp(params_df.loc["const", "coefficient"]), round_value)} + {exp_partial_formula}"
        

    def analyse_model_summary(summary_table: pd.DataFrame, top_count: int = 5, 
                              parameters: dict[str, bool] = {"aic_akaike_information_criterion": True,
                                                             "bic_bayesin_information_criterion": True,
                                                             "pseudo_r_2": False}) -> pd.DataFrame:
        # Generate an empty pandas.DataFrame
        temp_summary = summary_table.copy(deep = True).reset_index()
        temp_df = pd.DataFrame()
        # To sort dataframe according to parameters, where key is columns name and value is the sort ascending true or false
        # Then use head(top_count) to select the top count of dataframe that having the columns value
        # Reset index to know which model can be use to select the best model.
        # The dataframe is concated and then drop duplicates
        for key, value in parameters.items():
            temp_df = pd.concat([temp_df, temp_summary.sort_values(key, ascending = value).head(top_count)], 
                                ignore_index = True)\
                        .drop_duplicates()
            
        return temp_df

    def selecting_best_model(model_summary:pd.DataFrame, 
                             best_index:int, mode:str, 
                             dependent_variable: str,
                             columns_to_display:list = None) -> None:
        # Based on the best index, get the independent variables
        independent_variable = model_summary.loc[best_index, "variables"]
        if mode == "logistic":
            print(f"""From the summary table above, the {mode} regression with variable of \
{independent_variable} seems to be the best choice among all due to its \
AIC ({model_summary.loc[best_index, "aic_akaike_information_criterion"]}) is fairly low \
(lowest = {model_summary.loc[:,"aic_akaike_information_criterion"].min()}).
The BIC of the model ({model_summary.loc[best_index, "bic_bayesin_information_criterion"]}) is not too far higher than the minimum among the model summary of \
{model_summary.loc[:,"bic_bayesin_information_criterion"].min()}.
The psedo-r2 of the model showing that {model_summary.loc[best_index, "pseudo_r_2"]} of the {dependent_variable} can be \
explained by the independent variable of {independent_variable}.
The log_likelihood of {model_summary.loc[best_index, "log_likelihood"]} is not too high compare to the lowest \
{model_summary.loc[:,"log_likelihood"].min()}. 
The log_likelihood p value for the model is less than 0.05 \
indicating the model is indicating that the full model significantly improves the fit compared to the null model.
The model is having area under the Receiver Operating Characteristic (ROC) of {model_summary.loc[best_index, "roc"]}, \
which indicate the model is good to predict the {dependent_variable} based on its independent variables of {independent_variable}.
The Lagrange_Multiplier of {model_summary.loc[best_index, "Lagrange_Multiplier"]} with its p_value of \
{model_summary.loc[best_index, "Lagrange_Multiplier_p-value"]} which is less than 0.05,
along with f-statistic value of {model_summary.loc[best_index, "F-statistic"]} and f-statistic p value of \
{model_summary.loc[best_index, "F-statistic_p-value"]} indicating to reject the hypothesis of homodescacity where \
the residual variance does depend on the variables in x. Hence the model is having heterocesdacity in residual analysis of the model.""")
        
        elif mode == "poisson" or mode == "negative binominal" or mode == "zero inflated poisson" or mode == "zero inflated negative binominal":
            # print(f"From the summary table above, the {mode} regression with the variable of {independent_variable} seems to be the best choice among all due to:")
            print(f"For the {mode} regression model:")
            # print(model_summary.loc[:,columns_to_display].round(4).to_markdown(tablefmt = "pretty"))
            print(f"From the summary table above, the {mode} regression with the variable of {independent_variable} seems to be the best choice among all due to:")
            # AIC
            if model_summary.loc[best_index, "aic_akaike_information_criterion"] == model_summary.loc[:,"aic_akaike_information_criterion"].min():
                print(f"""Its AIC value of {model_summary.loc[best_index, "aic_akaike_information_criterion"]} is the lowest among all the models""")
            else:
                print(f"""Its AIC value of {model_summary.loc[best_index, "aic_akaike_information_criterion"]} is faily low as compare to lowest = {model_summary.loc[:,"aic_akaike_information_criterion"].min()}.""")
            # BIC
            if model_summary.loc[best_index, "bic_bayesin_information_criterion"] == model_summary.loc[:,"bic_bayesin_information_criterion"].min():
                print(f"""The BIC of the model ({model_summary.loc[best_index, "bic_bayesin_information_criterion"]}) is the lowest among all the models""")
            else:
                print(f"""The BIC of the model ({model_summary.loc[best_index, "bic_bayesin_information_criterion"]}) is faily low as compare to lowest = {model_summary.loc[:,"bic_bayesin_information_criterion"].min()}.""")
            # Dispersion Statistic
            if model_summary.loc[best_index, "dispersion_stats"] > 1.2:
                print(f"""However, by dividing the residual_pearson^2 by degree of freedom for residual values is {model_summary.loc[best_index, "dispersion_stats"]} more than 1, indicating overdispersion of the model.""")
            elif model_summary.loc[best_index, "dispersion_stats"] < 0.8:
                print(f"""However, by dividing the residual_pearson^2 by degree of freedom for residual values is {model_summary.loc[best_index, "dispersion_stats"]} less than 1, indicating underdispersion of the model.""")
            else:
                print(f"""In view of dividing the residual_pearson^2 by degree of freedom for residual values is {model_summary.loc[best_index, "dispersion_stats"]} almost equal to 1, indiciating the model is equidispersed.""")
            print("=======================================================================================")
            model_summary.loc[best_index].model.get_diagnostic().plot_probs().suptitle(f"Probability plot of {independent_variable} vs {dependent_variable}")

    def analyse_params(model, 
                       mode:str,
                       data_dictionary : dict, 
                       dependent_variable:str, 
                       round_value: int = 4) -> None:
        # Generate the params table
        params_df, exp_params_df = regression.generate_params_df(model)
        params_df, exp_params_df = params_df.round(round_value), exp_params_df.round(round_value)
        print("The params of the model =")
        print(params_df.to_markdown(tablefmt = "pretty"))
        print("The exponented params of the model =")
        print(exp_params_df.to_markdown(tablefmt = "pretty"))

        if mode == "logistic_regression":
            # Generate a temporary list for coefficients
            coefficient_list = set([column for column in params_df.index if column != 'const'])
            # To generate partial formula
            coefficient_list = list(zip(coefficient_list, 
                                        params_df.loc[params_df.index != "const","coefficient"], 
                                        exp_params_df.loc[exp_params_df.index != "const","coefficient"]))
            params_partial_formula = " + ".join([f"({round(value[1], round_value)} * {value[0]})" for value in coefficient_list])
            params_partial_formula = f"""{round(params_df.loc["const", "coefficient"], round_value)} + {params_partial_formula}"""
            exp_partial_formula = " + ".join([f"({value[2]:.2f} * {value[0]})" for value in coefficient_list])
            exp_partial_formula = f"""{round(np.exp(params_df.loc["const", "coefficient"]), round_value)} + {exp_partial_formula}"""

            # Prepare significant and not significant coefficient
            significant_coefficient = exp_params_df.loc[exp_params_df.loc[:,"p_value"] <= 0.05]
            not_significant_coefficient = exp_params_df.loc[exp_params_df.loc[:,"p_value"] > 0.05]

            # Print the formula
            print(f"The formula for the logistic model is Logit(p/(1-p)) = {params_partial_formula}.")
            print(f"The formula for the logistic model can be expressed in exponential value, which is p/(1-p) = {exp_partial_formula}.")
            print("===============================================================================")

            # To analyze based on coefficient
            # For constant
            print("Interpretation of Coefficients:")
            print(f"""The coefficient for constant is {exp_params_df.loc["const", "coefficient"]}, along with p_value of {exp_params_df.loc["const", "p_value"]}, which is significant.""")

            # Interpret the coefficients
            for index, row in exp_params_df.loc[exp_params_df.index != "const"].iterrows():
                print(f"{index}:")
                print(f"""For {data_dictionary[index][1]}, the odds of having {data_dictionary[dependent_variable][1]} is increased by \
    {row["coefficient"]} unit, holding all other variables constant.""")
                if row["p_value"] > 0.05:
                    print(f"""However, the confident interval for the {index} (95%CI = {row["2.5%"]}, {row["97.5%"]}) include 0 and its p value of {row["p_value"]} \
    is more than 0.05, therefore the coefficient of {index} not statiscally significant.""")
                else:
                    print(f"""The confident interval for the {index} (95%CI = {row["2.5%"]}, {row["97.5%"]}) did not include 0 and its p value of {row["p_value"]} \
    is less than 0.05, therefore the coefficient of {index} statiscally significant.""")
                print("===============================================================================")

        elif mode == "count":
            # Prepare the inflated part for zero inflated model
            inflated_const = "inflate_const"
            if inflated_const in params_df.index:
                inflated_list = [item for item in params_df.index if item.startswith('inflate_') and item != inflated_const]
                inflated_summary = list(zip(inflated_list, 
                                            params_df.loc[params_df.index.isin(inflated_list),"coefficient"], 
                                            exp_params_df.loc[exp_params_df.index.isin(inflated_list),"coefficient"]))
                                
                inflated_params_formula = " + ".join([f"({round(value[1], round_value)} * {value[0]})" for value in inflated_summary])
                inflated_params_formula = f"""{round(params_df.loc[inflated_const, "coefficient"], round_value)} + {inflated_params_formula}"""
                inflated_exp_params_formula = " + ".join([f"({round(value[2], round_value)} * {value[0]})" for value in inflated_summary])
                inflated_exp_params_formula = f"""{round(exp_params_df.loc[inflated_const, "coefficient"], round_value)} + {inflated_exp_params_formula}"""

                # print the logit part
                print("For inflate component, the formula as below:")
                print(f"Let 𝜋𝑖 is the probability of {dependent_variable} being a structural zero.")
                print(f"Logit(𝜋𝑖) = {inflated_params_formula}")
                # Check for alpha for negative binominal
                if "alpha" in params_df.index: 
                    print(f"""Where 𝒚𝒊~𝑵𝒆𝒈𝑩𝒊𝒏(𝝁𝒊, {round(params_df.loc["alpha", "coefficient"], round_value)}), 𝒗𝒂𝒓 𝒚𝒊 = 𝝁𝒊 + {round(params_df.loc["alpha", "coefficient"], round_value)}𝝁𝟐𝒊""")
                print(f"or if exponentialed it will become:")
                print(f"𝜋𝑖 = {inflated_exp_params_formula}")
                if "alpha" in params_df.index:
                    print(f"""Where 𝒚𝒊~𝑵𝒆𝒈𝑩𝒊𝒏(𝝁𝒊, {round(exp_params_df.loc["alpha", "coefficient"], round_value)}), 𝒗𝒂𝒓 𝒚𝒊 = 𝝁𝒊 + {round(exp_params_df.loc["alpha", "coefficient"], round_value)}𝝁𝟐𝒊""")
                print("-------------------------------------------------------------------------------")

                # To intepret the inflated coefficients
                print("Interpretation of Inflated Coefficients:")
                print(f"""The coefficient for inflated constant is log({round(params_df.loc[inflated_const, "coefficient"], round_value)}), \
or after exponential = {round(exp_params_df.loc[inflated_const, "coefficient"], round_value)} \
along with p_value of {round(exp_params_df.loc[inflated_const, "p_value"], round_value)}, \
which is {["significant" if exp_params_df.loc[inflated_const, "p_value"] < 0.05 else "not significant"]}.""")
                print("---------------------------")
                
                # to loop through each of the coefficients
                if len(inflated_list) > 0:
                    for index, row in params_df.loc[inflated_list].iterrows():
                        # For usage in data_dictionary
                        dict_index = index.replace('inflate_', '')
                        print(f"For odds of {index}:")
                        # If the key in data_dictionary indicate it is categorical type of data
                        if dict_index in list(data_dictionary.keys()):
                            print(f"""The odds of {data_dictionary[dict_index][1]} is associated with {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
in the odds of {dependent_variable} by a factor of {round(params_df.loc[index, "coefficient"], round_value)} or by \
{round(exp_params_df.loc[index, "coefficient"], round_value)} after exponentialed compared with {data_dictionary[dict_index][0]} \
when accounting for the counts via a Poisson model with the predictor {index}.""")
                        
                    else: # The index is continous type of data
                        print(f"""The odds of every single unit increase in {dict_index} is associated with {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
in the odds of {dependent_variable} by a factor of {round(params_df.loc[index, "coefficient"], round_value)} or by \
{round(exp_params_df.loc[index, "coefficient"], round_value)} after exponentialed\
when accounting for the counts via a Poisson model with the predictor {index}.""")
                    
                    # To print the 95% CI no matter what 
                    print(f"""The 95% of confidence interval = {round(params_df.loc[index, "2.5%"], round_value)} to \
{round(params_df.loc[index, "97.5%"], round_value)} with p value of \
{round(params_df.loc[index, "p_value"], round_value)}, which indicate there is a \
{["strong" if params_df.loc[index, "p_value"] < 0.05 else "weak"]} chance that {index} {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
of {dependent_variable} for every increase of unit in {index}.""")
                    # print a separator
                    print("---------------------------")
            
            # For poisson or binominal part
            not_inflated_const = "const"
            not_inflated_list = [item for item in params_df.index if item != not_inflated_const and not item.startswith('inflate_')]
            not_inflated_summary = list(zip(not_inflated_list, 
                                            params_df.loc[params_df.index.isin(not_inflated_list),"coefficient"], 
                                            exp_params_df.loc[exp_params_df.index.isin(not_inflated_list),"coefficient"]))

            not_inflated_params_formula = " + ".join([f"({round(value[1], round_value)} * {value[0]})" for value in not_inflated_summary])
            not_inflated_params_formula = f"""{round(params_df.loc[not_inflated_const, "coefficient"], round_value)} + {not_inflated_params_formula}"""
            not_inflated_exp_params_formula = " + ".join([f"({round(value[2], round_value)} * {value[0]})" for value in not_inflated_summary])
            not_inflated_exp_params_formula = f"""{round(exp_params_df.loc[not_inflated_const, "coefficient"], round_value)} + {not_inflated_exp_params_formula}"""

            # print the poisson part
            print("===============================================================================")
            print("For poisson component, the formula as below:")
            print(f"Let 𝜇𝑖 is the mean of {dependent_variable}.")
            print(f"Log(𝜇𝑖) = {not_inflated_params_formula}")
            if "alpha" in params_df.index: 
                print(f"""Where 𝒚𝒊~𝑵𝒆𝒈𝑩𝒊𝒏(𝝁𝒊, {round(params_df.loc["alpha", "coefficient"], round_value)}), 𝒗𝒂𝒓 𝒚𝒊 = 𝝁𝒊 + {round(params_df.loc["alpha", "coefficient"], round_value)}𝝁𝟐𝒊""")
            print(f"or if exponentialed it will become:")
            print(f"𝜇𝑖 = {not_inflated_exp_params_formula}")
            if "alpha" in params_df.index: 
                print(f"""Where 𝒚𝒊~𝑵𝒆𝒈𝑩𝒊𝒏(𝝁𝒊, {round(exp_params_df.loc["alpha", "coefficient"], round_value)}), 𝒗𝒂𝒓 𝒚𝒊 = 𝝁𝒊 + {round(exp_params_df.loc["alpha", "coefficient"], round_value)}𝝁𝟐𝒊""")
            print("-------------------------------------------------------------------------------")

            # Poisson coefficient
            print("Interpretation of not inflated Coefficients:")
            # For constant
            if "const" in params_df.index:
                print(f"""The coefficient for constant is {round(params_df.loc[not_inflated_const, "coefficient"], round_value)}, \
along with p_value of {round(exp_params_df.loc[not_inflated_const, "p_value"], round_value)}, \
which is {["significant" if exp_params_df.loc[not_inflated_const, "p_value"] < 0.05 else "not significant"]}.""")
                print("---------------------------")
            # Poisson part- IRR = exp(-0.0031)= 0.997: an age difference of 1 year is associated with a decrease of the mean number of difficulties with ADLs by a factor of 0.997 (or: by 0.3 %), once the structural zeroes have been modelled separately and are predicted by age.
            # But 95 % CI: 0.993 – 1.000, p = 0.084, and there is no age effect.

            # To loop through the index
            for index, row in params_df.loc[not_inflated_list].iterrows():
                print(f"For incidence rate of {index}:")
                if index in list(data_dictionary.keys()):
                    print(f"""The {data_dictionary[index][1]} is associated with {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
of the mean number of {dependent_variable} by a factor of {round(params_df.loc[index, "coefficient"], round_value)} or by \
{round(exp_params_df.loc[index, "coefficient"], round_value)} after exponentialed compared with {data_dictionary[index][0]} \
once the structural zeroes have been modelled separately and are predicted by {index}.""")
                    
                else:
                    print(f"""A difference of every single unit increase in {index} is associated with {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
of the mean number of {dependent_variable} by a factor of {round(params_df.loc[index, "coefficient"], round_value)} or by \
{round(exp_params_df.loc[index, "coefficient"], round_value)} after exponentialed\
once the structural zeroes have been modelled separately and are predicted by {index}.""")
                
                print(f"""The 95% of confidence interval = {round(params_df.loc[index, "2.5%"], round_value)} to \
{round(params_df.loc[index, "97.5%"], round_value)} with p value of \
{round(params_df.loc[index, "p_value"], round_value)}, which indicate there is a \
{["strong" if params_df.loc[index, "p_value"] < 0.05 else "weak"]} chance that {index} {["increase" if params_df.loc[index, "coefficient"] > 0 else "decrease"]} \
of {dependent_variable} for every increase of unit in {index}.""")
                # Print a separator
                print("---------------------------")

    def analyse_model(model, 
                      df:pd.DataFrame, 
                      mode:str, 
                      data_dictionary : dict, 
                      dependent_variable:str, 
                      round_value: int = 4):
        # print the summary table at first
        print(model.summary())

        # Analyse the params
        regression.analyse_params(model = model, 
                                  mode = mode,
                                  data_dictionary=data_dictionary, 
                                  dependent_variable=dependent_variable, 
                                  round_value=round_value)
        
        # For VIF if independent variable more than 2
        independent_variables = set([column for column in model.params.index 
                                                                         if column != 'const' and 
                                                                         column != 'alpha' and 
                                                                         not column.startswith('inflate_')])
        
        # For multicollinearity
        if len(independent_variables) > 1:
            vif_data = regression.calculate_vif(df = df, independent_variables = independent_variables)
            regression.analyse_vif(vif_data=vif_data, dependent_variable=dependent_variable, round_value=round_value)

        # For model fit
        # The llr 
        LLR_p_value = round(model.llr_pvalue, round_value)
        log_likelihood_value = round(model.llf, round_value)
        LL_Null_value = round(model.llnull, round_value)
        if LLR_p_value < 0.05:
            print(f"""This logistic model having {log_likelihood_value = }, {LL_Null_value = }.
The {LLR_p_value = } indicating that the full model significantly improves the fit compared to the null model.""")
        else:
            print(f"""This logistic model having {log_likelihood_value = }, {LL_Null_value = }.
The {LLR_p_value = } indicating that the full model is not significantly improves the fit compared to the null model.""")

        # for AIC & BIC
        aic = model.aic
        bic = model.bic
        print(f"The Akaike Information Criterion (AIC) of the model = {aic} and Bayesin Information Criterion = {bic}.")

        # Based on model will be different
        if mode == "logistic_regression":
            # To draw ROC
            fpr, tpr, roc_auc = regression.plot_roc_auc(model = model, df = df, dependent_variable=dependent_variable,
                                                            data_dictionary=data_dictionary, round_value = round_value)
            
            # For Residual Plot
            regression.plot_residual(model = model)

        if mode == "count":
            print("=================================================================================================================")
            # For model of fit, use pseudo r2, pearson goodness of fit test and deviance goodness of fit test, llr, aic
            print("For other model fitness:")
            # Pseudo r 2
            pseudo_r_2 = round(model.prsquared, round_value)
            print(f"The pseudo r2 for the model = {pseudo_r_2} which might indicate a mild lack of fit compared to poisson model.")
            print("=================================================================================================================")

            # Person goodness of fit test
            try:
                pearson_prob = model.get_diagnostic().test_chisquare_prob()
                print(f"""The pearson goodness of fit test having degree of freedom of {round(pearson_prob.df, round_value)}, \
with chi2 value of {round(pearson_prob.statistic, round_value)}, and p value of {round(pearson_prob.pvalue, round_value)}, which is \
{['p < 0.05, indicate' if pearson_prob.pvalue < 0.05 else 'p >= 0.05, indicate not']} significant fit of the model based on {independent_variables} \
against frequency of {dependent_variable}""")
            except:
                print("")
            print("=================================================================================================================")
            
            # For converged
            if model.converged == True:
                print("The model's converged is True, indicating the model is considered reliable because the estimates have stabilized and (hopefully) represent the true underlying relationship in the data.")
            else:
                print("he model's converged is False, indicating the model is not considered to be reliable as the algorithm unable to reaches a point where the estimated coefficients no longer change significantly between iterations.")
            print("=================================================================================================================")
            
            # To test on overdispersion
            try:
                dispersion_table = model.get_diagnostic().test_dispersion().summary_frame()
                print(f"Table of dispersion test for poisson regression with {independent_variables}:")
                print(dispersion_table.round(round_value).to_markdown(tablefmt = "pretty", index = False))
                if dispersion_table.loc[:,"pvalue"].mean() < 0.05:
                    print("All the p value < 0.05 test, indicate we reject the null hypothesis. The model is not in equidispersion.")
                else:
                    print("Not all the p value < 0.05, we failed ot reject the null hypothesis. The model might be in equidispersion")
            except:
                dispersion_statistic = (model.resid_pearson**2).sum() / model.df_resid
                if dispersion_statistic > 1.2:
                    print(f"However, by dividing the residual_pearson^2 by degree of freedom for residual values is {dispersion_statistic} more than 1, indicating overdispersion of the model.")
                elif dispersion_statistic < 0.8:
                    print(f"However, by dividing the residual_pearson^2 by degree of freedom for residual values is {dispersion_statistic} less than 1, indicating underdispersion of the model.")
                else:
                    print(f"In view of dividing the residual_pearson^2 by degree of freedom for residual values is {dispersion_statistic} almost equal to 1, indiciating the model is equidispersed.")
            print("=================================================================================================================")
            
            
            
            # Plot the graph
            model.get_diagnostic().plot_probs().suptitle(f"Probability plot of {independent_variables} vs {dependent_variable}")


    def plot_roc_auc(model, 
                     df:pd.DataFrame, 
                     dependent_variable:str, 
                     data_dictionary:dict, 
                     round_value:int = 4) -> tuple[np.array, np.array, float]:
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        params_df, exp_params_df = regression.generate_params_df(model)

        independent_variable = [column for column in params_df.index if column != 'const']

        # Step 1: Obtain predicted probabilities
        y_pred_prob = model.predict(sm.add_constant(df[independent_variable]))

        # Step 2: Plot the ROC curve
        fpr, tpr, thresholds = roc_curve(df.loc[:,dependent_variable], y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        print(f"The Receiver Operating Characteristics (ROC) and Area under the ROC Curve for the model is {round(roc_auc, round_value)}.")

        if roc_auc > 0.9:
            print(f"The model has Excellent performance at distinguishing between {data_dictionary[dependent_variable][0]} and {data_dictionary[dependent_variable][1]}")
        elif 0.8 <= roc_auc < 0.9:
            print(f"The model has Very good performance at distinguishing between {data_dictionary[dependent_variable][0]} and {data_dictionary[dependent_variable][1]}")
        elif 0.7 <= roc_auc < 0.8:
            print(f"The model has Good performance at distinguishing between {data_dictionary[dependent_variable][0]} and {data_dictionary[dependent_variable][1]}")
        elif 0.6 <= roc_auc < 0.7:
            print(f"The model has Satisfactory performance at distinguishing between {data_dictionary[dependent_variable][0]} and {data_dictionary[dependent_variable][1]}")
        else:
            print(f"The model has Unsatisfactory performance at distinguishing between {data_dictionary[dependent_variable][0]} and {data_dictionary[dependent_variable][1]}")

        # Plot the graph
        # plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {round(roc_auc, round_value)})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) for {independent_variable}')
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

        # Return the fpr, tpr from roc, and roc_auc as well
        return fpr, tpr, roc_auc
    
    def plot_residual(model) -> None:
        import matplotlib.pyplot as plt
        # To obtain independent variables
        params_df, exp_params_df = regression.generate_params_df(model)

        independent_variable = [column for column in params_df.index if column != 'const']

        # Obtain predicted probabilities
        predicted_probs = model.predict()

        # Compute residuals
        residuals = model.resid_response

        # Generate figure with 4 axis
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))

        # Diagnostic plotting
        # Residual plot
        ax1.scatter(predicted_probs, residuals, alpha=0.8)
        ax1.set_xlabel("Predicted Probabilities")
        ax1.set_ylabel("Residuals")
        ax1.set_title(f"Residual Plot for \n{independent_variable}")
        ax1.axhline(y=0, color='r', linestyle='--')

        # Histogram of residuals
        ax2.hist(residuals, bins=20, edgecolor='k')
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Histogram of Residuals of \n{independent_variable}")

        # Q-Q plot
        sm.qqplot(residuals, line='45', ax = ax3)
        ax3.set_title(f"Q-Q Plot of Residuals of \n{independent_variable}")

        # Homoscedasticity check
        ax4.scatter(predicted_probs, residuals**2, alpha=0.8)
        ax4.set_xlabel("Predicted Probabilities")
        ax4.set_ylabel("Squared Residuals")
        ax4.set_title(f"Homoscedasticity Check for \n{independent_variable}")
        ax4.axhline(y=0, color='r', linestyle='--')

        plt.show()
        plt.close()

    def calculate_vif(df:pd.DataFrame, independent_variables:list|str|tuple) -> pd.DataFrame:
        from statsmodels.tools.tools import add_constant
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        # Checking on type of independent variables if given value
        column_name = pre_processing.identify_independent_variable(independent_variables)
        # Calculate VIF
        X_with_const = add_constant(df.loc[:,column_name])
        
        # Calculate Condition Index
        corr_matrix = df.loc[:,column_name].corr()
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        condition_index = np.sqrt(eigenvalues.max() / eigenvalues)

        vif_data = pd.DataFrame({"feature":list(independent_variables),
                                 "VIF":[variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])],
                                 "condition_index":condition_index})
        
        return vif_data
    
    def analyse_vif(vif_data:pd.DataFrame, dependent_variable:str, round_value:int = 4) -> None:
        vif_data = vif_data.round(round_value)
        # Print the VIF
        print("Below showing the variance inflaction factors (VIF) for each of the independent variable of the model.")
        print(vif_data.to_markdown(tablefmt = "pretty"))

        for index, row in vif_data.iterrows():
            if row["feature"] == "const":
                continue
            if row["VIF"] >= 10:
                print(f"Variable '{row['feature']}' has a VIF of {row['VIF']} and condition index of {row['condition_index']}, indicating significant multicollinearity for this {dependent_variable}.")
            elif row["VIF"] >= 5:
                print(f"Variable '{row['feature']}' has a VIF of {row['VIF']} and condition index of {row['condition_index']}, indicating potential multicollinearity that may be problematic for this {dependent_variable}.")
            else:
                print(f"Variable '{row['feature']}' has a VIF of {row['VIF']} and condition index of {row['condition_index']}, indicating no significant multicollinearity for this {dependent_variable}.")

        if len(vif_data.loc[vif_data.loc[:,"VIF"] >= 10]) > 0:
            print(f"""There is independent variable that have VIF value more than 10 against {dependent_variable}, which are {list(vif_data.loc[vif_data.loc[:,"VIF"] >= 10, 'feature'])}.""")
        elif len(vif_data.query("VIF >= 5 and VIF < 10")) > 0:
            print(f"""There is independent variable that have VIF value more than 5 but less than 10 against {dependent_variable}, which are {list(vif_data.query("VIF >= 5 and VIF < 10").loc[:, 'feature'])}.""")
        elif (len(vif_data.query("VIF < 5"))) >0:
            print(f"""The independent variable of {list(vif_data.query("VIF < 5").loc[:,'feature'])} shows no significant multicollinearity for {dependent_variable}.""")

    def crude_analysis(summary_model_df:pd.DataFrame, 
                       df:pd.DataFrame,
                       mode:str,
                       dependent_variable:str,
                       data_dictionary:dict,
                       round_value:int = 4,
                       p_value_cut_off:float = 0.05) -> pd.DataFrame:
        # Prepare empty dataframe
        combine_params_df = pd.DataFrame()
        combine_exp_params_df = pd.DataFrame()
        # To select only variable with num_variables == 1
        for index in summary_model_df.query("num_variables == 1").index:
            # Preparing the model
            model =summary_model_df.loc[index, "model"]
            print(f"""Crude analysis for {summary_model_df.loc[index, "variables"]}:""")
            regression.analyse_model(model = model, 
                                     df = df, 
                                     mode = mode,
                                     dependent_variable=dependent_variable, 
                                     round_value=round_value, 
                                     data_dictionary=data_dictionary)
            print("============================================================================================")

            # Generate params df
            params_df, exp_params_df = regression.generate_params_df(model)
            independent_variable = summary_model_df.loc[index, "variables"]

            combine_params_df = pd.concat([combine_params_df, params_df.loc[(independent_variable, "const"),]], ignore_index = False)
            combine_exp_params_df = pd.concat([combine_exp_params_df, exp_params_df.loc[(independent_variable, "const"),]], ignore_index = False)

        # Drop index == const
        combine_params_df = combine_params_df.loc[combine_params_df.index != "const"]
        combine_exp_params_df = combine_exp_params_df.loc[combine_exp_params_df.index != "const"]
        print("The odds for all independent variables:")
        print(combine_exp_params_df.round(round_value).to_markdown(tablefmt = "pretty"))

        # Determine coefficient significant or not based on p_value_cut_off
        significant_coefficients = [index for index in combine_exp_params_df.index if 
                                    combine_exp_params_df.loc[index, "p_value"] < p_value_cut_off and 
                                    index != "const" and index != "alpha" and not index.startswith('inflate_')]
        not_significatn_coefficients = [index for index in combine_exp_params_df.index if 
                                    combine_exp_params_df.loc[index, "p_value"] >= p_value_cut_off and
                                    index != "const" and index != "alpha" and not index.startswith('inflate_')]
        
        if len(significant_coefficients) > 0:
            print(f"""From the table above, {significant_coefficients} is having p value less than {p_value_cut_off}, which indicate the coefficients are statistically significant.""")
        if len(not_significatn_coefficients) > 0:
            print(f"""From the table above, {not_significatn_coefficients} is having p value more than {p_value_cut_off}, which indicate the coefficients are not statistically significant.""")


        return combine_params_df, combine_exp_params_df
        
    def generate_adjusted_crude_params(summary_df:pd.DataFrame, model_index:int,
                                       round_value:int = 4,
                                       p_value_cut_off:float = 0.05) -> pd.DataFrame:
        summary_params_df = pd.DataFrame()
        summary_exp_params_df = pd.DataFrame()
        
        model = summary_df.loc[model_index].model

        adj_params_df, adj_exp_params_df = regression.generate_params_df(model)

        for params in model.params.index[1:]:
            index = summary_df.loc[summary_df.loc[:,"variables"] == params].index.values[0]
            params_df, exp_params_df = regression.generate_params_df(summary_df.loc[index].model)
            
            summary_params_df = pd.concat([summary_params_df, params_df.iloc[1:]], ignore_index=False)
            summary_exp_params_df = pd.concat([summary_exp_params_df, exp_params_df.iloc[1:]], ignore_index=False)

        summary_exp_params_df = summary_exp_params_df.loc[:,("coefficient", "p_value", "2.5%", "97.5%")]\
                                                     .merge(adj_exp_params_df.loc[:,("coefficient", "p_value", "2.5%", "97.5%")], 
                                                            how = "outer", left_index=True, right_index=True,
                                                            suffixes=["","_adj"])
        
        summary_params_df = summary_params_df.loc[:,("coefficient", "p_value", "2.5%", "97.5%")]\
                                             .merge(adj_params_df.loc[:,("coefficient", "p_value", "2.5%", "97.5%")], 
                                                    how = "outer", left_index=True, right_index=True,
                                                    suffixes=["","_adj"])
        # Return both table
        return summary_params_df, summary_exp_params_df
    
    def vuong_test (m1, m2):
        from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
        from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
        from scipy.stats import norm

        supported_models = [ZeroInflatedPoisson,
                            ZeroInflatedNegativeBinomialP,
                            Poisson,
                            NegativeBinomial]

        if type (m1.model) not in supported_models:
            raise ValueError (f"""Model type not supported for first \
parameter. List of supported models: \
(ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, \
Poisson, NegativeBinomial) from statsmodels discrete \
collection.""")

        if type (m2.model) not in supported_models:
            raise ValueError (f"Model type not supported for second \
parameter. List of supported models: \
(ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP, \
Poisson, NegativeBinomial) from statsmodels discrete \
collection.")

        m1_y = m1.model.endog
        m2_y = m2.model.endog

        m1_n = len(m1_y)
        m2_n = len(m2_y)

        if m1_n == 0 or m2_n == 0:
            raise ValueError ("Could not extract dependent variables from \
models.")

        if m1_n != m2_n:
            raise ValueError ("Models appear to have different numbers of \
observations.\n" \
f"Model 1 has {m1_n} observations.\n" \
f"Model 2 has {m2_n} observations.")

        if np.any(m1_y != m2_y):
            raise ValueError ("Models appear to have different values on dependent variables.")

        m1_linpred = pd.DataFrame (m1.predict (which ="prob"))
        m2_linpred = pd.DataFrame (m2.predict (which ="prob"))

        m1_probs = np.repeat (np.nan, m1_n)
        m2_probs = np.repeat (np.nan, m2_n)

        which_col_m1 = [list (m1_linpred.columns).index (x) if x in
                        list (m1_linpred.columns) else None for x in m1_y]
        which_col_m2 = [list(m2_linpred.columns).index (x) if x in
                        list(m2_linpred.columns) else None for x in m2_y]

        for i, v in enumerate (m1_probs):
            m1_probs [i] = m1_linpred.iloc [i, which_col_m1[i]]

        for i, v in enumerate (m2_probs):
            m2_probs [i] = m2_linpred.iloc [i, which_col_m2[i]]

        lm1p = np.log(m1_probs)
        lm2p = np.log(m2_probs)

        m = lm1p - lm2p

        v = np.sum(m) / (np.std (m) * np.sqrt(len(m)))

        pval = 1 - norm.cdf(v) if v > 0 else norm.cdf(v)

        print ("Vuong Non - Nested Hypothesis Test - Statistic (Raw):")
        print (f"Vuong z- statistic: {v}")
        print (f"p-value: {pval}")


    def generate_count_regression_combine_table(model_list:list, independent_variables:list,
                                                best_model_index:int, model_name:list,) ->pd.DataFrame:
        # Generate empty list
        exp_params_list = []
        # To look for inflation
        inflation_independent = [f"inflate_{variable}" for variable in independent_variables]
        # Loop through the model list
        for model in model_list:
            # Generate the params and exponential params
            params_df, exp_params_df = regression.generate_params_df(model[10])
            # Drop unnecessary features
            exp_params_df = exp_params_df.drop(columns = ["t_statistic", "2.5%", "97.5%"])
            # Add features that might be useful
            exp_params_df.loc["converged", "coefficient"] = model[best_model_index].converged
            exp_params_df.loc["aic", "coefficient"] = model[best_model_index].aic
            exp_params_df.loc["bic", "coefficient"] = model[best_model_index].bic
            exp_params_df.loc["pseudo_r_2", "coefficient"] = model[best_model_index].prsquared
            exp_params_df.loc["log_likelihood", "coefficient"] = model[best_model_index].llf
            exp_params_df.loc["ll_null", "coefficient"] = model[best_model_index].llnull
            exp_params_df.loc["llr_p_value", "coefficient"] = model[best_model_index].llr_pvalue
            exp_params_df.loc["dispersion_statistic", "coefficient"] = (model[best_model_index].resid_pearson**2).sum() / model[best_model_index].df_resid

            # Append into the list
            exp_params_list.append(exp_params_df)

        # Return the concatenated dataframe
        return pd.concat(exp_params_list, axis = 1, keys= model_name)\
                .reindex(index = ["const",] +  list(independent_variables) + 
                        ["inflation", "inflate_const",] + inflation_independent + 
                        ["alpha", "converged", "aic", "bic", "pseudo_r_2", "log_likelihood", "ll_null", 
                         "llr_p_value", "dispersion_statistic",])
    

    def find_confounder(summary_df:pd.DataFrame, 
                        fixed_independent_variable:str, 
                        round_value:int = 4) -> pd.DataFrame:
        # Get the B1 formula
        b1 = summary_df.query(f"variables == '{fixed_independent_variable}'").loc[:,f"coefficient_{fixed_independent_variable}"].values[0]
        # Get all the model with variables at least contain fixed_independent_variable
        temp = summary_df.query("num_variables == 2")
        temp = temp.loc[temp.loc[:,"variables"].str.contains(fixed_independent_variable)]

        # Create empty dataframe and list
        confounder_summary_table = pd.DataFrame()
        confounder_list = []
        not_confounder_list = []
        # To get the model
        for index, row in temp.iterrows():
            params = row["model"].params
            b2_variable = [variable for variable in params.index 
                            if variable != "const" and variable != fixed_independent_variable
                            and variable != "alpha"][0]
        
            # Calculate the change of coefficients
            percentage_changes = ((np.abs(b1 - params.loc[fixed_independent_variable]))/ np.abs(b1)) * 100

            if percentage_changes > 10:
                confounder_list.append(b2_variable)
            else:
                not_confounder_list.append(b2_variable)

            # Concatenate the dataframe
            confounder_summary_table = pd.concat([confounder_summary_table, pd.DataFrame({f"b1_{fixed_independent_variable}":[b1],
                                                                                          f"b1_p_value":[summary_df.query(f"variables == '{fixed_independent_variable}'").loc[:,f"p_value_{fixed_independent_variable}"].values[0]],
                                                                                          f"b2_variables_{fixed_independent_variable}+":[b2_variable],
                                                                                          f"b2_{fixed_independent_variable}":[params.loc[fixed_independent_variable]],
                                                                                          "b1-b2/b1":[percentage_changes]})],
                                ignore_index = True)
            
        # Print the analysis
        print(confounder_summary_table.round(round_value).to_markdown(tablefmt = "pretty"))
        # Statement based on the len of list
        if len(confounder_list) > 0:
            print(f"""Noted coefficient of {fixed_independent_variable} changed more than 10% if model including {confounder_list} as shown on table above, indicating the {confounder_list} is possible confounder for the model above.""")
        else:
            print("No confounde factors was found in the model.")
        if len(not_confounder_list) > 0:
            print(f"""The coefficients of {not_confounder_list} did not change the b1 of {fixed_independent_variable} for more than 10%, indicating the variable of {not_confounder_list} not the confounder factors for the model.""")
        
        # Return the summary table
        return confounder_summary_table