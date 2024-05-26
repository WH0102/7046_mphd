# k-nearestneighbour(kNN)
# Supportvectormachine(SVM)
# Logisticregression(LR)
# Decisiontree(DT)
# Randomforest(RF)
# Extremegradientboost(XGB)
# Lightgradientboosting machine (LGBM)

import pandas as pd

class machine_learning:
    def automl(X_train:pd.DataFrame, y_train:pd.DataFrame, **kwargs):
        # https://microsoft.github.io/FLAML/docs/Use-Cases/task-oriented-automl#customize-automlfit
        # Import Necessary Packages
        from flaml import AutoML
        automl = AutoML()

        # Sample kwargs
        settings = {
            "time_budget": 30,  # total running time in seconds
            "metric": 'r2',  # primary metrics for regression can be chosen from: ['mae','mse','r2','rmse','mape']
            "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
            "task": 'regression',  # task type    
            "log_file_name": 'automl.log',  # flaml log file
            "seed": 7654321,    # random seed
            "use_spark": False,  # whether to use Spark for distributed training
            "n_concurrent_trials": 2,  # the maximum number of concurrent trials
        }

        # To fit the automl
        automl.fit(X_train=X_train, y_train=y_train, **kwargs)

        # Return the model
        return automl

    def LightGBM(df:pd.DataFrame):
        # Import the necessary packages
        from lightgbm import LGBMClassifier