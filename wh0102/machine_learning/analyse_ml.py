import pandas as pd

class analyse_ml:
    def analyse_automl(automl, X_test:pd.DataFrame, y_test:pd.DataFrame):
        from flaml.ml import sklearn_metric_loss_score
        import matplotlib.pyplot as plt
        
        print('Best hyperparmeter config:', automl.best_config)
        print('Best r2 on validation data: {0:.4g}'.format(1-automl.best_loss))
        print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

        y_pred = automl.predict(X_test)
        print('r2', '=', 1 - sklearn_metric_loss_score('r2', y_pred, y_test))
        print('mse', '=', sklearn_metric_loss_score('mse', y_pred, y_test))
        print('mae', '=', sklearn_metric_loss_score('mae', y_pred, y_test))

        # Plot barh
        plt.barh(automl.feature_names_in_, automl.feature_importances_)

    def analyse_sklearn():
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report