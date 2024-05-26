# k-nearestneighbour(kNN)
# Supportvectormachine(SVM)
# Logisticregression(LR)
# Decisiontree(DT)
# Randomforest(RF)
# Extremegradientboost(XGB)
# Lightgradientboosting machine (LGBM)

import pandas as pd
from numpy import linspace

class machine_learning:
    def automl(X_train:pd.DataFrame, y_train:pd.DataFrame, **kwargs):
        # https://microsoft.github.io/FLAML/docs/Use-Cases/task-oriented-automl#customize-automlfit
        # https://github.com/microsoft/FLAML/blob/main/notebook/automl_classification.ipynb
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

    def LightGBM(X_train:pd.DataFrame,
                 y_train:str|list|tuple|None,
                 independent_variables_continous:str|list|tuple|None,
                 n_estimators:list = [100, 300],
                 max_depth:list = [3,4,5],
                 min_child_weight = range(1,3,1),
                 learning_rate = [0.1,0.2],
                 booster = ['gbdt'],
                 reg_alpha = [1e-5,0.01,0.03],
                 weights = linspace(0.3, 0.9, 2),
                 num_leaves = [6],
                 scoring:str = "roc_auc",
                 n_jobs:int = -1,
                 n_splits_for_lgbm:int = 5,
                 random_seed:int|None = None):
        # Import the necessary packages
        from ..pre_processing.pre_processing import pre_processing
        from lightgbm import LGBMClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        import time
    
        # Params
        # n_estimators = [100, 300]
        # max_depth = [3,4,5]
        # min_child_weight = range(1,3,1)
        # booster = ['gbdt']
        # base_score = [0.5,0.6]
        # learning_rate = [0.1,0.2]
        # objective = ['binary']
        # seed = [27]
        # gamma= [0.7,0.8,0.9]
        # colsample_bytree=[0.7,0.8,0.9]
        # subsample=[0.6,0.7,0.8]
        # reg_alpha = [1e-5,0.01,0.03]
        # weights = linspace(0.3, 0.9, 2)
        # num_leaves = [6]

        lgbm_params = {'classifier__n_estimators': n_estimators, 
                       'classifier__max_depth': max_depth,
                       'classifier__learning_rate' : learning_rate, 
                       'classifier__min_child_weight' : min_child_weight, 
                       'classifier__boosting_type' : booster, 
                       'smote__sampling_strategy': weights,
                       'classifier__reg_alpha':reg_alpha, 
                       'classifier__num_leaves':num_leaves}
               
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[('num', RobustScaler(), independent_variables_continous),], 
            remainder='passthrough'
        )

        pipeline = Pipeline([('smote', SMOTE(random_state=random_seed)),
                             ('scaler', preprocessor),
                             ('classifier', LGBMClassifier())])

        stratified_kfold = StratifiedKFold(n_splits=n_splits_for_lgbm,
                                           shuffle=True,
                                           random_state=random_seed)
        
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=lgbm_params,
                                   scoring=scoring,
                                   cv=stratified_kfold,
                                   n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
