# k-nearestneighbour(kNN)
# Supportvectormachine(SVM)
# Logisticregression(LR)
# Decisiontree(DT)
# Randomforest(RF)
# Extremegradientboost(XGB)
# Lightgradientboosting machine (LGBM)

import pandas as pd

class machine_learning:
    def LightGBM(df:pd.DataFrame):
        # Import the necessary packages
        from lightgbm import LGBMClassifier