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

    def analyse_ml(model,
                   model_type:str,
                   time_required:float,
                   independent_variables:str|list|tuple|None,
                   X_test:pd.DataFrame,
                   y_test:pd.DataFrame) -> pd.DataFrame:
        # Import Necessary packages
        import numpy as np
        from pprint import pprint
        from sklearn.metrics import (
            roc_curve, roc_auc_score,
            accuracy_score, precision_score, recall_score, f1_score, classification_report)
        from sklearn.feature_selection import RFE
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Print Best Params
        print("Best Params:")
        pprint(model.best_params_)
        print("------------------------------------------------")

        # Get the best estimator from the GridSearchCV object
        best_estimator = model.best_estimator_

        # extract the XGBClassifier from the pipeline
        xgb_clf = best_estimator.named_steps["classifier"]

        # Variance due to method
        if model_type == "Decision Tree" or model_type == "Random Forest" or model_type =="XGB" or model_type == "LightGBM":
            # get the feature importances
            importances = xgb_clf.feature_importances_
        elif model_type == "Logistic Regression":
            # get the feature importances
            importances = list(xgb_clf.coef_[0])
        elif model_type == "kNN" or model_type == "SVM":
            from sklearn.inspection import permutation_importance
            importances = permutation_importance(model, X_test, y_test).importances_mean

        # create a DataFrame with feature importances and feature names as columns
        importance_df = pd.DataFrame(data={'feature_names': independent_variables, 
                                        'importances': importances})\
                        .sort_values(by='importances', ascending=False,)

        # Create a bar chart of feature importances
        plt.figure(figsize=(12,6))
        plt.bar(x=np.arange(importance_df.shape[0]), height=importance_df['importances'])
        plt.xticks(np.arange(importance_df.shape[0]), importance_df['feature_names'], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        plt.show()
        plt.close()

        # ROC Curve
        # Get the predicted probabilities for the test set
        try:
            y_test_proba = best_estimator.predict_proba(X_test)[:, 1]
        except:
            y_test_proba = best_estimator.predict(X_test)

        # Compute the fpr, tpr, and thresholds for the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

        # Compute the AUC
        auc = roc_auc_score(y_test, y_test_proba)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.annotate(f'AUC = {auc:.4f}', xy=(0.8, 0.2), xycoords='axes fraction')
        plt.legend(loc='best')
        plt.show()
        plt.close()

        # Define a list of threshold values to check
        thresholds = np.linspace(0.0005, 1, 1000)

        # Create empty lists to store the results
        sensitivities = []
        specificities = []
        accuracies = []
        youden_indices = []

        # Iterate over the threshold values
        for threshold in thresholds:
            # Modify the predicted probabilities based on the threshold
            y_test_pred = [1 if prob >= threshold else 0 for prob in y_test_proba]

            # Compute the confusion matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred)

            # Extract true positives, true negatives, false positives, and false negatives
            tp = conf_matrix[1, 1]
            tn = conf_matrix[0, 0]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]

            # Calculate sensitivity, specificity, and accuracy
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Append the results to the lists
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)

            # Calculate Youden's Index and append it to the list
            youden_index = sensitivity + specificity - 1
            youden_indices.append(youden_index)

        # Find the threshold that maximizes Youden's Index
        optimal_idx = np.argmax(youden_indices)
        optimal_threshold = thresholds[optimal_idx]
        max_youden_index = youden_indices[optimal_idx]

        # Extract corresponding sensitivity, specificity, and accuracy for the optimal threshold
        optimal_sensitivity = sensitivities[optimal_idx]
        optimal_specificity = specificities[optimal_idx]
        optimal_accuracy = accuracies[optimal_idx]

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, sensitivities, label='Sensitivity')
        plt.plot(thresholds, specificities, label='Specificity')
        plt.plot(thresholds, accuracies, label='Accuracy')

        # Mark the optimal threshold with a vertical dashed line
        plt.axvline(x=optimal_threshold, color='red', linestyle='--')
        plt.text(optimal_threshold, 0.0, f'Optimal Threshold: {optimal_threshold:.4f}', ha='left', color='red')
        plt.text(optimal_threshold, optimal_sensitivity - 0.05, f'Sensitivity: {optimal_sensitivity:.4f}', ha='left', color='blue')
        plt.text(optimal_threshold, optimal_specificity - 0., f'Specificity: {optimal_specificity:.4f}', ha='left', color='orange')
        plt.text(optimal_threshold, optimal_accuracy - 0.05, f'Accuracy: {optimal_accuracy:.4f}', ha='left', color='green')

        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics Across Different Thresholds')
        plt.show()

        # Dataframe for confusion matrix at optimal threshold
        y_test_pred = [1 if prob >= optimal_threshold else 0 for prob in y_test_proba]
        y_test_pred_str = ["Correct_Prediction" if prob >= optimal_threshold else "Wrong_Prediction" for prob in y_test_proba]
        # Compute the confusion matrix
        conf_matrix = pd.crosstab(y_test, y_test_pred_str, 
                                rownames=['Condition'], 
                                margins=True)\
                        .rename(index = {0:"False Condition",
                                        1:"True Condition"})
                                

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix.to_markdown(tablefmt = "pretty"))

        # Print classifiction report
        print("Classification Report")
        print(classification_report(y_test, y_test_pred))

        # Summary of model
        summary_df = pd.DataFrame({"model_type":[model_type],
                                   "time_required":[time_required],
                                   "accuracy_score":[accuracy_score(y_test, y_test_pred)],
                                   "precision_score":[precision_score(y_test, y_test_pred)],
                                   "recall_score":[recall_score(y_test, y_test_pred)],
                                   "f1_score":[f1_score(y_test, y_test_pred)],
                                   "cv_score":[model.best_score_],
                                   "test_score":[model.score(X_test, y_test)],
                                   "auc":[auc],
                                   "optimal_threshold":[optimal_threshold],
                                   "optimal_sensitivity":[optimal_sensitivity],
                                   "optimal_specificity":[optimal_specificity],
                                   "optimal_accuracy":[optimal_specificity],
                                   "max_youden_s_index":[max_youden_index]})
        
        return summary_df