import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, shapiro
import statsmodels.api as sm

class continous_data:
    def descriptive_analysis(df: pd.DataFrame, round:int = 2, cut_of_value:int = 10):
        # Using pandas default describe
        descriptive_df = df.describe()

        # Calculate the skew and kurtosis
        for formula in [skew, kurtosis]:
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
        
        # Print the descriptive table
        if round != None:
            print(descriptive_df.round(round).to_markdown(tablefmt = "pretty"))
        else:
            print(descriptive_df.to_markdown(tablefmt = "pretty"))

        # Store a list of normal distribution data
        normal_distribution_list = [column for column in descriptive_df.columns 
                                    if descriptive_df.loc["shapiro_p_value", column] >= 0.05]
        abnormal_distribution_list = [column for column in descriptive_df.columns 
                                      if descriptive_df.loc["shapiro_p_value", column] < 0.05]
        
        # Plot QQ graph if have normally distributed data
        if len(normal_distribution_list) > 0:
            for column in normal_distribution_list:
                print(f"QQ plot for {column}")
                sm.qqplot(df.loc[:,column], line = "45", fit = True)
        return descriptive_df