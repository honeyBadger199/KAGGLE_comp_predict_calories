import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_manipulation import data_manager




class PrepareModel:
    def __init__(self):
        pass
    def loss_function(self, y_true, y_pred):
        return 'rmsle',np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred))),False
    def train_models(self,X,Y,x_test,folds=5):
        lgb_predictions = np.zeros((len(x_test)))
        cat_predictions = np.zeros((len(x_test)))
        kF =KFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kF.split(X, Y)):
            X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
            X_val,y_val = X.iloc[val_idx], Y.iloc[val_idx]
            lgbm = LGBMRegressor(objective = 'regression',
                                  boosting_type = 'gbdt',
        
                                  n_estimators = 1000,
                                  learning_rate = 0.01,
                                  num_leaves = 31,
                                  max_depth = -1,
                                  min_child_samples = 20,
                                  subsample = 0.8,
                                  colsample_bytree = 0.8,
                                  random_state = 42)
            cat = CatBoostRegressor(iterations=1000,
                                     learning_rate=0.01,
                                     depth=6,
                                     l2_leaf_reg=3,
                                     loss_function='RMSLE',
                                     eval_metric='RMSLE',
                                     random_seed=42)
            lgbm.fit(X_train, Y_train,
                    eval_set=[(X_train,Y_train), (X_val, y_val)],
                    eval_metric=self.loss_function)
            lgb_predictions += lgbm.predict(x_test) / folds
            cat.fit(X_train, Y_train,
                    eval_set=[(X_train,Y_train), (X_val, y_val)])
            cat_predictions += cat.predict(x_test) / folds
        return lgb_predictions, cat_predictions
    def ensemble(self, lgb_predictions, cat_predictions):
        ensemble_preds = (lgb_predictions + cat_predictions) / 2
        return np.expm1(ensemble_preds)
        
class Pipeline:
    def __init__(self):
        self.data_manager_train = data_manager.DataManager("/home/navneet/Kaggle_comp/playground-series-s5e5/train.csv")
        self.data_manager_test = data_manager.DataManager("/home/navneet/Kaggle_comp/playground-series-s5e5/test.csv")
        self.submission_data = data_manager.DataManager("/home/navneet/Kaggle_comp/playground-series-s5e5/sample_submission.csv")
    def run(self):
        training_data = self.data_manager_train.load_data()
        training_data = self.data_manager_train.add_new_features(training_data)
        training_data = self.data_manager_train.tartget_transform(training_data)
        test_data = self.data_manager_test.load_data()
        test_data = self.data_manager_test.add_new_features(test_data)
        submission_data = self.submission_data.load_data()
        print("Data loaded successfully.")
        numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
        categorical_features = ['Sex']
        X = training_data[numerical_features + categorical_features]
        y = training_data['Calories'] 
        x_test = test_data[numerical_features+categorical_features]
        prepare_model_and_train = PrepareModel()
        lgb_prediction,cat_predictions = prepare_model_and_train.train_models(X,y,x_test)
        print("Models trained successfully.")
        final_predictions = prepare_model_and_train.ensemble(lgb_prediction,cat_predictions)
        submission_data['Calories'] = final_predictions
        submission_data.to_csv('submission.csv', index=False)
        print("Submission file created successfully.")
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()