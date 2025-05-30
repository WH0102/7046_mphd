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
                 params:dict,
                 independent_variables_continous:str|list|tuple|None = None,
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

        # Preprocessing
        independent_variables = pre_processing.identify_independent_variable(independent_variables_continous)

        # Transform
        preprocessor = ColumnTransformer(
                transformers=[('num', RobustScaler(), independent_variables),], 
                remainder='passthrough'
            )

        pipeline = Pipeline([('smote', SMOTE(sampling_strategy='auto', random_state=random_seed)),
                             ('scaler', preprocessor),
                             ('classifier', LGBMClassifier())])

        stratified_kfold = StratifiedKFold(n_splits=n_splits_for_lgbm,
                                           shuffle=True,
                                           random_state=random_seed)
        
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=params,
                                   scoring=scoring,
                                   cv=stratified_kfold,
                                   n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def random_forest(X_train:pd.DataFrame,
                      y_train:str|list|tuple|None,
                      params:dict,
                      scoring:str = "roc_auc",
                      n_jobs:int = -1,
                      n_splits:int = 5,
                      random_seed:int|None = None):
        # Import the necessary packages
        from ..pre_processing.pre_processing import pre_processing
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        import time

        pipeline = Pipeline(steps = [['smote', SMOTE(random_state=random_seed)],
                                     ['classifier', RandomForestClassifier()]])


        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                           shuffle=True,
                                           random_state=random_seed)
        
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=params,
                                   scoring=scoring,
                                   cv=stratified_kfold,
                                   n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def logisticRegression(X_train:pd.DataFrame,
                           y_train:str|list|tuple|None,
                           params:dict,
                           scoring:str = "roc_auc",
                           n_jobs:int = -1,
                           n_splits:int = 5,
                           random_seed:int|None = None):
        # Import the necessary packages
        from sklearn.linear_model import LogisticRegression
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        import time

        # Define the pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=random_seed)),
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression())
        ])

        # Create the StratifiedKFold object
        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                            shuffle=True,
                                            random_state=random_seed)
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=pipeline,
                                    param_grid=params,
                                    scoring=scoring,
                                    cv=stratified_kfold,
                                    n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def knn(X_train:pd.DataFrame,
            y_train:str|list|tuple|None,
            params:dict,
            scoring:str = "roc_auc",
            n_jobs:int = -1,
            n_splits:int = 5,
            random_seed:int|None = None):
        # Import the necessary packages
        from sklearn.neighbors import KNeighborsClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        import time

        # Define the pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=random_seed)),
            ('scaler', RobustScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # Create the StratifiedKFold object
        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                            shuffle=True,
                                            random_state=random_seed)
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=pipeline,
                                    param_grid=params,
                                    scoring=scoring,
                                    cv=stratified_kfold,
                                    n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def svm(X_train:pd.DataFrame,
            y_train:str|list|tuple|None,
            params:dict,
            scoring:str = "roc_auc",
            n_jobs:int = -1,
            n_splits:int = 5,
            random_seed:int|None = None):
        # Import the necessary packages
        from sklearn.svm import SVC
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        import time

        # Define the pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=random_seed)),
            ('scaler', RobustScaler()),
            ('classifier', SVC(probability=True))
        ])

        # Create the StratifiedKFold object
        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                            shuffle=True,
                                            random_state=random_seed)
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=pipeline,
                                    param_grid=params,
                                    scoring=scoring,
                                    cv=stratified_kfold,
                                    n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def decision_tree(X_train:pd.DataFrame,
                      y_train:str|list|tuple|None,
                      params:dict,
                      scoring:str = "roc_auc",
                      n_jobs:int = -1,
                      n_splits:int = 5,
                      random_seed:int|None = None):
        # Import the necessary packages
        from sklearn.tree import DecisionTreeClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        import time

        # Define the pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=random_seed)),
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier())
        ])

        # Create the StratifiedKFold object
        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                            shuffle=True,
                                            random_state=random_seed)
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=pipeline,
                                    param_grid=params,
                                    scoring=scoring,
                                    cv=stratified_kfold,
                                    n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    
    def xgb(X_train:pd.DataFrame,
            y_train:str|list|tuple|None,
            params:dict,
            scoring:str = "roc_auc",
            n_jobs:int = -1,
            n_splits:int = 5,
            random_seed:int|None = None):
        # Import the necessary packages
        from xgboost import XGBClassifier
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        import time

        # Define the pipeline
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=random_seed)),
            ('scaler', RobustScaler()),
            ('classifier', XGBClassifier())
        ])

        # Create the StratifiedKFold object
        stratified_kfold = StratifiedKFold(n_splits=n_splits,
                                            shuffle=True,
                                            random_state=random_seed)
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(estimator=pipeline,
                                    param_grid=params,
                                    scoring=scoring,
                                    cv=stratified_kfold,
                                    n_jobs=n_jobs)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        time_required = end_time - start_time
        print(f'Time taken: {end_time - start_time:.3f} seconds')

        return grid_search, time_required
    