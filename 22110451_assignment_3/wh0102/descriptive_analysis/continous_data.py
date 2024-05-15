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

        # independent_columns = tuple([column for column in independent_columns if len(df.loc[:,column].unique()) >= cut_off_value])

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
            plt.figure(figsize=(6, 4))

            # Draw heatmap for correlation matrix
            sns.heatmap(df.loc[:,all_variables].corr(), annot=True, cmap="coolwarm", linewidths=.5, fmt=".3f")
            plt.title('Correlation Matrix')

            plt.show()

        # Close the plot
        plt.close()
