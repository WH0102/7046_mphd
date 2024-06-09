import pandas as pd
import numpy as np

import statsmodels.api as sm
from ..pre_processing.pre_processing import pre_processing


class continous_data:
    def descriptive_analysis(df: pd.DataFrame, 
                             independent_variables : list|tuple|set|str = None, 
                             dependent_variables: str = None,
                             descriptive_type: str = "continous",
                             plot_dependent_variables: bool = True,
                             plot_correlation: bool = True,
                             round:int = 2,
                             cut_off_value:int = 10) -> tuple[list, list]:
        from scipy.stats import skew, kurtosis, shapiro, norm, spearmanr
        # To check on independent_variables with cut_off_value to ensure it is continous data
        independent_columns = pre_processing.identify_independent_variable(independent_variables)

        independent_columns = tuple([column for column in independent_columns if len(df.loc[:,column].unique()) >= cut_off_value])

        # Using pandas default describe
        descriptive_df = df.describe()

        # Calculate the skew and kurtosis
        for formula in [np.var, skew, kurtosis]:
            descriptive_df.loc[f"{formula.__name__}"] = [formula(df.loc[:,column]) 
                                                         if df.loc[:,column].dtype == int or df.loc[:,column].dtype == float 
                                                         else None 
                                                         for column in descriptive_df.columns]

        # Check for normal distribution
        # Calculate the shapiro
        shapiro_values = [shapiro(df.loc[:,column])
                          if df.loc[:,column].dtype == int or df.loc[:,column].dtype == float 
                          else None 
                          for column in descriptive_df.columns]
        
        # Put the shapiro and its p_value into the descriptive df
        descriptive_df.loc["shapiro"] = [shapiro_value[0] for shapiro_value in shapiro_values]
        descriptive_df.loc["shapiro_p_value"] = [shapiro_value[1] for shapiro_value in shapiro_values]
      
        if descriptive_type == "distribution":
            spearman_correlation = [spearmanr(df.loc[:,column], df.loc[:,dependent_variables])
                                    if df.loc[:,column].dtype == int or df.loc[:,column].dtype == float 
                                    else None 
                                    for column in descriptive_df.columns]
            descriptive_df.loc["spearmanr"] = [value[0] for value in spearman_correlation]
            descriptive_df.loc["spearmanr_p_value"] = [value[1] for value in spearman_correlation]
        
        # Print the descriptive table
        if round != None:
            print(descriptive_df.round(round).to_markdown(tablefmt = "pretty"))
        else:
            print(descriptive_df.to_markdown(tablefmt = "pretty"))

        # Store a list of normal distribution data
        normal_distribution_list = [column for column in independent_columns
                                    if descriptive_df.loc["shapiro_p_value", column] >= 0.05]
        abnormal_distribution_list = [column for column in independent_columns
                                      if descriptive_df.loc["shapiro_p_value", column] < 0.05]
        
        # print some statement
        if descriptive_type == "continous":
            continous_data.plot_continous_data(df = df, 
                                               independent_variables = independent_columns, 
                                               dependent_variables = dependent_variables,
                                               plot_dependent_variables = plot_dependent_variables,
                                               plot_correlation = plot_correlation)
        elif descriptive_type == "distribution":
            # generate pivot table
            temp_pt = df.pivot_table(index = dependent_variables, values = independent_columns[0], aggfunc=len, margins=True)\
                        .rename(columns = {independent_columns[0]:"count"})
            # Calculate the percentage
            temp_pt.loc[:,"percentage"] = temp_pt.loc[:,"count"] / temp_pt.loc["All", "count"]

            # print the pivot table
            if round != None:
                print(temp_pt.round(round).to_markdown(tablefmt = "pretty"))
            else:
                print(temp_pt.to_markdown(tablefmt = "pretty"))

            # plot the histogram
            continous_data.plot_continous_data(df = df, 
                                               independent_variables = independent_columns, 
                                               dependent_variables = dependent_variables,
                                               plot_mode = "distribution",
                                               plot_dependent_variables = plot_dependent_variables,
                                               plot_correlation = plot_correlation)

        # Return both columns category in distribution manner
        return normal_distribution_list, abnormal_distribution_list

    def plot_continous_data(df:pd.DataFrame, 
                            independent_variables : list|tuple|set|str = None, 
                            dependent_variables: str = None,
                            plot_mode:str = "continous",
                            plot_dependent_variables: bool = True,
                            plot_correlation: bool = True) -> None:
        # Import important packages
        from matplotlib import pyplot as plt
        import seaborn as sns

        # Checking on type of independent variables if given value
        independent_columns = pre_processing.identify_independent_variable(independent_variables)
        
        if dependent_variables != None:
            dependent_variables = (dependent_variables,)
            all_variables = independent_columns + dependent_variables
        else:
            all_variables = independent_columns

        # Print a statement
        print("Plotting graphs for independent variables")
        
        # Plot for independent variable
        for  column_name in independent_columns:
            # Prepare the figure based on availability of dependent variables
            if dependent_variables != None:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(18, 3))
            else:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))

            # Histogram
            ax1.hist(df.loc[:,column_name], bins=len(df.loc[:,column_name].unique()), edgecolor='black')
            ax1.set_title(f'Histogram for {column_name}')
            ax1.set_xlabel(column_name)
            ax1.set_ylabel('Frequency')

            # QQ plot
            sm.qqplot(df.loc[:,column_name], ax=ax2, line='45', fit = True)
            ax2.set_title(f'QQ Plot for {column_name}')

            # Box plot
            sns.boxplot(df.loc[:,column_name], ax=ax3)
            ax3.set_title(f'Box Plot for {column_name}')

            if dependent_variables != None:
                # Scatter plot
                ax4.scatter(df.loc[:,column_name], df.loc[:,dependent_variables])
                ax4.set_title(f'Scatter plot of {column_name} vs {dependent_variables}')
                ax4.set_xlabel(column_name)
                ax4.set_ylabel(dependent_variables)

            # Adjust spacing between subplots
            plt.subplots_adjust(wspace=0.3)
            
            # Display the plot
            plt.show()

        if plot_dependent_variables == True:
            if plot_mode == "continous":
                if dependent_variables != None:
                    # Print a statement
                    print("Plotting graphs for dependent variables")

                    # Plot for dependent variable
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))

                    # Histogram len(df.loc[:,dependent_variables].unique()) -> DataFrame have no unique
                    ax1.hist(df.loc[:,dependent_variables], bins=20, edgecolor='black')
                    ax1.set_title(f'Histogram for dependent variable of {dependent_variables}')
                    ax1.set_xlabel(dependent_variables)
                    ax1.set_ylabel('Frequency')

                    # QQ plot
                    sm.qqplot(df.loc[:,dependent_variables], ax=ax2, line='45', fit = True)
                    ax2.set_title(f'QQ Plot for {dependent_variables}')

                    # Box plot
                    sns.boxplot(df.loc[:,dependent_variables], ax=ax3)
                    ax3.set_title(f'Box Plot for {dependent_variables}')

                    # Adjust spacing between subplots
                    plt.subplots_adjust(wspace=0.3)
                                    
                    # Display the plot
                    plt.show()
            elif plot_mode == "distribution":
                import plotly.express as px
                px.histogram(df, x = dependent_variables[0], 
                             text_auto = True, 
                             title = "Distribution of number of {dependent_variables}")
                

        # Plot for correlation based on plot_correlation value
        if plot_correlation == True:
            print("Plotting correlation matrix")
            # Set up the matplotlib figure
            plt.figure(figsize=(10, 6))

            # Draw heatmap for correlation matrix
            sns.heatmap(df.loc[:,all_variables].corr(), annot=True, cmap="coolwarm", linewidths=.5, fmt=".3f")
            plt.title('Correlation Matrix')

            plt.show()

        # Close the plot
        plt.close()

    def identify_outliers(df:pd.DataFrame, 
                          column_name:str|list|tuple, 
                          ratio:float=1.5,
                          normal_values:dict=None,
                          handle_outliers:str=None):
        """
        This function identifies outliers in a pandas DataFrame.

        Args:
            data: The pandas DataFrame containing the data.
            column_name: The name of the column to identify outliers in.
            ratio: The number of standard deviations or interquatile ratio to define an outlier (default: 1.5).
            normal_values: Dictionary that contain normal values.

        Returns:
            A string containing information about the outliers, 
            including the number of outliers, their indices, and their values.
        """
        # Import necessary packages
        from scipy.stats import skew, kurtosis, shapiro, norm, spearmanr, iqr

        # To tuple the colmns
        columns = pre_processing.identify_independent_variable(column_name)

        # Prepare empty dataframe
        outliers_df = pd.DataFrame()

        # To loop through the columns
        for column_name in columns:
            # Calculate shapiro
            if shapiro(df.loc[:,column_name])[1] >= 0.05:
                # Calculate the mean and standard deviation of the column, 
                mean = df.loc[:,column_name].mean()
                std = df.loc[:,column_name].std()
                # Define the upper and lower bounds for outliers
                upper_bound = mean + (ratio * std)
                lower_bound = mean - (ratio * std)
            else:
                median = df.loc[:,column_name].median()
                upper_bound = median + (ratio * iqr(df.loc[:,column_name]))
                lower_bound = median - (ratio * iqr(df.loc[:,column_name]))

            # To set based on normal value as well
            if normal_values != None:
                if column_name in normal_values.keys():
                    upper_bound = upper_bound if upper_bound > normal_values[column_name][1] else normal_values[column_name][1]
                    lower_bound = lower_bound if lower_bound < normal_values[column_name][0] else normal_values[column_name][0]

            # Identify outliers based on the bounds
            outliers = df.loc[(df.loc[:,column_name] < lower_bound) | (df.loc[:,column_name] > upper_bound)]\
                         .sort_values(column_name, ascending = False)

            # To proceed based on outliers length
            if len(outliers) > 0:
                # Concatenate with outliers_df
                outliers_df = pd.concat([outliers_df, outliers], ignore_index=True).drop_duplicates()

                # Print information about the outliers
                print(f"Outliers found in column of {column_name} with boundaries of {lower_bound:,.2f} - {upper_bound:,.2f}:")
                if normal_values != None:
                    if column_name in normal_values.keys():
                        print(f"Normal Value : {normal_values[column_name][0]} - {normal_values[column_name][1]}")
                # Print the dataset
                print(outliers.to_markdown(tablefmt = "pretty"))
                print("---------------------------------------------------------------")
            
            # To handle outliers
                if handle_outliers == "cap":
                    df.loc[(df.loc[:,column_name] < lower_bound), column_name] = lower_bound.astype(df.loc[:,column_name].dtypes)
                    df.loc[(df.loc[:,column_name] > upper_bound), column_name] = upper_bound.astype(df.loc[:,column_name].dtypes)

        if len(outliers_df) > 0:
            print(f"The overall dataframe have total of {len(outliers_df)} ({len(outliers_df)/len(df)*100:.2f}%) found:")
            print(outliers_df.to_markdown(tablefmt = "pretty"))
        else:
            print("Overall no outliers found in the dataframe with selected columns of {columns}")

        # Return a dataframe
        return outliers_df
