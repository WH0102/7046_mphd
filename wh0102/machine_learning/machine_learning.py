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
        from flaml import AutoML
        automl = AutoML()  

        # To fit the automl
        automl.fit(X_train=X_train, y_train=y_train, **kwargs)

        # Return the model
        return automl

    def LightGBM(df:pd.DataFrame):
        # Import the necessary packages
        from lightgbm import LGBMClassifier